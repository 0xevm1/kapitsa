#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/gpu.h"
#include "util/log.h"

#include <mutex>
#include <unordered_map>

namespace ni {
extern const char* const kKernelSource;

struct Gpu::Impl {
  std::mutex mu;
  std::unordered_map<std::string, Pipeline> pipelines;
  NSMutableArray* retained = [NSMutableArray new]; // keeps MTLBuffers alive
};

static inline id<MTLDevice> dev(void* p) { return (__bridge id<MTLDevice>)p; }

Gpu& Gpu::get() {
  static Gpu* g = new Gpu();
  return *g;
}

Gpu::Gpu() : impl_(new Impl) {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (!d) fatal("no Metal device");
  device_ = (__bridge_retained void*)d;
  id<MTLCommandQueue> q = [d newCommandQueueWithMaxCommandBufferCount:64];
  queue_ = (__bridge_retained void*)q;
}

std::string Gpu::name() const { return [[dev(device_) name] UTF8String]; }
size_t Gpu::recommended_working_set() const { return dev(device_).recommendedMaxWorkingSetSize; }

Buffer Gpu::alloc(size_t bytes, const char* label, bool tracked) {
  if (bytes == 0) bytes = 16;
  id<MTLBuffer> b = [dev(device_) newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared |
                                                      (tracked ? MTLResourceHazardTrackingModeTracked
                                                               : MTLResourceHazardTrackingModeUntracked)];
  if (!b) fatal("Metal alloc of %zu bytes failed", bytes);
  if (label) b.label = [NSString stringWithUTF8String:label];
  std::lock_guard<std::mutex> lk(impl_->mu);
  [impl_->retained addObject:b];
  return Buffer{(__bridge void*)b, (uint8_t*)b.contents, 0, bytes};
}

Buffer Gpu::wrap(void* ptr, size_t bytes, const char* label) {
  id<MTLBuffer> b = [dev(device_) newBufferWithBytesNoCopy:ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared |
                                                           MTLResourceHazardTrackingModeUntracked
                                               deallocator:nil];
  if (!b) fatal("Metal no-copy wrap of %zu bytes failed (page aligned?)", bytes);
  if (label) b.label = [NSString stringWithUTF8String:label];
  std::lock_guard<std::mutex> lk(impl_->mu);
  [impl_->retained addObject:b];
  return Buffer{(__bridge void*)b, (uint8_t*)ptr, 0, bytes};
}

void Gpu::release(Buffer& b) {
  if (!b.mtl) return;
  std::lock_guard<std::mutex> lk(impl_->mu);
  [impl_->retained removeObjectIdenticalTo:(__bridge id)b.mtl];
  b = Buffer{};
}

void Gpu::compile(const std::string& extra) {
  if (library_) return;
  @autoreleasepool {
    std::string src = std::string(kKernelSource) + "\n" + extra;
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_1;
    if (@available(macOS 15.0, *)) {
      o.mathMode = MTLMathModeFast;
    } else {
      o.fastMathEnabled = YES;
    }
    NSError* err = nil;
    double t0 = now_ms();
    id<MTLLibrary> lib = [dev(device_) newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                                    options:o
                                                      error:&err];
    if (!lib) fatal("Metal compile failed: %s", err ? [[err description] UTF8String] : "?");
    library_ = (__bridge_retained void*)lib;
    logi("compiled kernels in %.0f ms", now_ms() - t0);
  }
}

Pipeline Gpu::pipeline(const std::string& fn) {
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->pipelines.find(fn);
    if (it != impl_->pipelines.end()) return it->second;
  }
  compile();
  @autoreleasepool {
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library_;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
    if (!f) fatal("kernel %s not found", fn.c_str());
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev(device_) newComputePipelineStateWithFunction:f error:&err];
    if (!pso) fatal("pipeline %s: %s", fn.c_str(), [[err description] UTF8String]);
    Pipeline p{(__bridge_retained void*)pso, (uint32_t)pso.maxTotalThreadsPerThreadgroup,
               (uint32_t)pso.threadExecutionWidth};
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->pipelines[fn] = p;
    return p;
  }
}

std::unique_ptr<Encoder> Gpu::begin(bool concurrent) {
  auto e = std::unique_ptr<Encoder>(new Encoder());
  id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;
  id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
  id<MTLComputeCommandEncoder> enc =
      [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
  e->cb_ = (__bridge_retained void*)cb;
  e->enc_ = (__bridge_retained void*)enc;
  return e;
}

void Gpu::make_resident(const std::vector<Buffer>& bufs) {
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      if (!residency_) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.label = @"nace-infer";
        NSError* err = nil;
        id<MTLResidencySet> rs = [dev(device_) newResidencySetWithDescriptor:d error:&err];
        if (!rs) return;
        residency_ = (__bridge_retained void*)rs;
        [(__bridge id<MTLCommandQueue>)queue_ addResidencySet:rs];
      }
      id<MTLResidencySet> rs = (__bridge id<MTLResidencySet>)residency_;
      for (auto& b : bufs)
        if (b.mtl) [rs addAllocation:(__bridge id<MTLBuffer>)b.mtl];
      [rs commit];
      [rs requestResidency];
    }
  }
}

// ---- Encoder ----
static inline id<MTLComputeCommandEncoder> E(void* p) { return (__bridge id<MTLComputeCommandEncoder>)p; }

void Encoder::use(const Pipeline& p) { [E(enc_) setComputePipelineState:(__bridge id<MTLComputePipelineState>)p.pso]; }
void Encoder::buf(int idx, const Buffer& b, size_t extra) {
  [E(enc_) setBuffer:(__bridge id<MTLBuffer>)b.mtl offset:b.offset + extra atIndex:idx];
}
void Encoder::bytes(int idx, const void* data, size_t len) { [E(enc_) setBytes:data length:len atIndex:idx]; }
void Encoder::tgmem(int idx, size_t len) { [E(enc_) setThreadgroupMemoryLength:(len + 15) & ~size_t(15) atIndex:idx]; }
void Encoder::dispatch(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  [E(enc_) dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
}
void Encoder::barrier() { [E(enc_) memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
void Encoder::commit() {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
    enc_ = nullptr;
  }
  [(__bridge id<MTLCommandBuffer>)cb_ commit];
}
void Encoder::wait() {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  [cb waitUntilCompleted];
  if (cb.status == MTLCommandBufferStatusError)
    fatal("GPU error: %s", [[cb.error description] UTF8String]);
}
double Encoder::gpu_ms() const {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cb_;
  return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}
Encoder::~Encoder() {
  if (enc_) {
    [E(enc_) endEncoding];
    CFRelease(enc_);
  }
  if (cb_) CFRelease(cb_);
}

} // namespace ni
