Streaming output truncated to the last 5000 lines.
  144 |         #pragma unroll 1
      | 
/content/STORM/cutlass/include/cutlass/barrier.h:160: warning: ignoring ‘#pragma unroll ’ [-Wunknown-pragmas]
  160 |         #pragma unroll 1
      | 
/content/STORM/cutlass/include/cutlass/barrier.h:174: warning: ignoring ‘#pragma unroll ’ [-Wunknown-pragmas]
  174 |         #pragma unroll 1
      | 
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp:339: warning: ignoring ‘#pragma unroll ’ [-Wunknown-pragmas]
  339 |         #pragma unroll
      | 
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:57,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:312: warning: ignoring ‘#pragma unroll ’ [-Wunknown-pragmas]
  312 |       #pragma unroll(IterationsUnroll ? kIterations : 1)
      | 
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:399: warning: ignoring ‘#pragma unroll ’ [-Wunknown-pragmas]
  399 |       #pragma unroll(IterationsUnroll ? kIterations : 1)
      | 
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeThreadIdxX()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:50:10: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread0’?
   50 |   return threadIdx.x;
      |          ^~~~~~~~~
      |          thread0
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeThreadIdxY()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:56:10: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread0’?
   56 |   return threadIdx.y;
      |          ^~~~~~~~~
      |          thread0
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeThreadIdxZ()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:62:10: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread0’?
   62 |   return threadIdx.z;
      |          ^~~~~~~~~
      |          thread0
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockIdxX()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:68:10: error: ‘blockIdx’ was not declared in this scope
   68 |   return blockIdx.x;
      |          ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockIdxY()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:74:10: error: ‘blockIdx’ was not declared in this scope
   74 |   return blockIdx.y;
      |          ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockIdxZ()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:80:10: error: ‘blockIdx’ was not declared in this scope
   80 |   return blockIdx.z;
      |          ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockDimX()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:86:10: error: ‘blockDim’ was not declared in this scope
   86 |   return blockDim.x;
      |          ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockDimY()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:92:10: error: ‘blockDim’ was not declared in this scope
   92 |   return blockDim.y;
      |          ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h: In function ‘int cutlass::gemm::threadblock::RematerializeBlockDimZ()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/index_remat.h:98:10: error: ‘blockDim’ was not declared in this scope
   98 |   return blockDim.z;
      |          ^~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h: In constructor ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::ThreadblockSwizzleStreamK(cutlass::gemm::GemmUniversalMode, cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord, int, int, int, int, size_t, size_t, size_t, int)’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:159:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::batch_count’ will be initialized after [-Wreorder]
  159 |   int batch_count;
      |       ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:144:7: warning:   ‘int cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::reduction_blocks’ [-Wreorder]
  144 |   int reduction_blocks;
      |       ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:144:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::reduction_blocks’ will be initialized after [-Wreorder]
  144 |   int reduction_blocks;
      |       ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:140:7: warning:   ‘int cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::dp_blocks’ [-Wreorder]
  140 |   int dp_blocks;                            /// Number of data-parallel thread blocks in the grid
      |       ^~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:149:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::sk_iters_per_region’ will be initialized after [-Wreorder]
  149 |   int sk_iters_per_region;
      |       ^~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:146:7: warning:   ‘int cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::sk_waves’ [-Wreorder]
  146 |   int sk_waves;
      |       ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:146:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::sk_waves’ will be initialized after [-Wreorder]
  146 |   int sk_waves;
      |       ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:135:7: warning:   ‘int cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::sm_occupancy’ [-Wreorder]
  135 |   int sm_occupancy;
      |       ^~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:135:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::sm_occupancy’ will be initialized after [-Wreorder]
  135 |   int sm_occupancy;
      |       ^~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:132:8: warning:   ‘bool cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::remap_block_indices’ [-Wreorder]
  132 |   bool remap_block_indices;
      |        ^~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:138:7: warning: ‘cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::avail_sms’ will be initialized after [-Wreorder]
  138 |   int avail_sms;
      |       ^~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:129:8: warning:   ‘bool cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::cohort_raster’ [-Wreorder]
  129 |   bool cohort_raster;
      |        ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:400:3: warning:   when initialized here [-Wreorder]
  400 |   ThreadblockSwizzleStreamK(
      |   ^~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h: In member function ‘int cutlass::gemm::threadblock::ThreadblockSwizzleStreamK::device_num_blocks() const’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:641:12: error: ‘gridDim’ was not declared in this scope
  641 |     return gridDim.x;
      |            ^~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm.h:42,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:43,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/semaphore.h: In member function ‘void cutlass::Semaphore::wait(int)’:
/content/STORM/cutlass/include/cutlass/semaphore.h:92:12: error: ‘__syncthreads_and’ was not declared in this scope
   92 |     while( __syncthreads_and(state != status) ) {
      |            ^~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/semaphore.h:96:5: error: ‘__syncthreads’ was not declared in this scope
   96 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/semaphore.h: In member function ‘void cutlass::Semaphore::release(int)’:
/content/STORM/cutlass/include/cutlass/semaphore.h:102:5: error: ‘__syncthreads’ was not declared in this scope
  102 |     __syncthreads();
      |     ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:43,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm.h: In member function ‘void cutlass::gemm::kernel::Gemm<Mma_, Epilogue_, ThreadblockSwizzle_, SplitKSerial>::operator()(const cutlass::gemm::kernel::Gemm<Mma_, Epilogue_, ThreadblockSwizzle_, SplitKSerial>::Params&, cutlass::gemm::kernel::Gemm<Mma_, Epilogue_, ThreadblockSwizzle_, SplitKSerial>::SharedStorage&)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm.h:238:22: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  238 |     int thread_idx = threadIdx.x;
      |                      ^~~~~~~~~
      |                      thread_idx
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/aligned_buffer.h: At global scope:
/content/STORM/cutlass/include/cutlass/aligned_buffer.h:83:9: error: declaration of ‘using Array = struct cutlass::Array<T, N>’ changes meaning of ‘Array’ [-fpermissive]
   83 |   using Array = Array<T, N>;
      |         ^~~~~
In file included from /content/STORM/cutlass/include/cutlass/fast_math.h:44,
                 from /content/STORM/cutlass/include/cutlass/layout/matrix.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:39,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/array.h:51:8: note: ‘Array’ declared here as ‘struct cutlass::Array<T, N>’
   51 | struct Array;
      |        ^~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/layout/permute.h: In member function ‘cutlass::layout::PermuteBase::LongIndex cutlass::layout::Tensor4DPermuteBMM0213RowMajor<D1>::operator()(cutlass::MatrixCoord) const’:
/content/STORM/cutlass/include/cutlass/layout/permute.h:317:27: error: ‘blockIdx’ was not declared in this scope
  317 |     Index BMM_batch_idx = blockIdx.z;
      |                           ^~~~~~~~
/content/STORM/cutlass/include/cutlass/layout/permute.h: In member function ‘cutlass::layout::PermuteBase::LongIndex cutlass::layout::Tensor4DPermuteBMM0213RowMajorInverse<D1>::operator()(cutlass::MatrixCoord) const’:
/content/STORM/cutlass/include/cutlass/layout/permute.h:380:27: error: ‘blockIdx’ was not declared in this scope
  380 |     Index BMM_batch_idx = blockIdx.z;
      |                           ^~~~~~~~
/content/STORM/cutlass/include/cutlass/layout/permute.h: In member function ‘cutlass::layout::PermuteBase::LongIndex cutlass::layout::Tensor4DPermuteBMM0321ColumnMajor<D1>::operator()(cutlass::MatrixCoord) const’:
/content/STORM/cutlass/include/cutlass/layout/permute.h:452:27: error: ‘blockIdx’ was not declared in this scope
  452 |     Index BMM_batch_idx = blockIdx.z;
      |                           ^~~~~~~~
/content/STORM/cutlass/include/cutlass/layout/permute.h: In member function ‘cutlass::layout::PermuteBase::LongIndex cutlass::layout::Tensor4DPermuteBMM0321ColumnMajorInverse<D1>::operator()(cutlass::MatrixCoord) const’:
/content/STORM/cutlass/include/cutlass/layout/permute.h:513:27: error: ‘blockIdx’ was not declared in this scope
  513 |     Index BMM_batch_idx = blockIdx.z;
      |                           ^~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h: At global scope:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h:87:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   87 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h:814:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  814 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator.h:1088:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorNCxHWx<InterleavedN> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1088 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorNCxHWx<InterleavedN> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h: In member function ‘void cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::reduce(int, int, int, void*, const OutputOp&, cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::OutputTileIterator, cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::OutputTileIterator)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:335:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  335 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:335:5: note: (if you use ‘-fpermissive’, G++ will accept your code, but allowing the use of an undeclared name is deprecated)
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h: In member function ‘void cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::unified(const OutputOp&, cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::OutputTileIterator, const AccumulatorTile&, cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::OutputTileIterator)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:424:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  424 |       __syncthreads();  // Dummy (CUDA 11.0)
      |       ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h: In member function ‘void cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::operator()(const OutputOp&, cutlass::epilogue::threadblock::Epilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_, FragmentsPerPartition, IterationsUnroll>::OutputTileIterator, const AccumulatorTile&, SourceAspect)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:494:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  494 |       __syncthreads();
      |       ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue.h:499:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  499 |       __syncthreads();
      |       ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/thread/linear_combination.h:41,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<float, signed char, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:5465:15: error: there are no arguments to ‘__dp4a’ that depend on a template parameter, so a declaration of ‘__dp4a’ must be available [-fpermissive]
 5465 |       t[ii] = __dp4a(x, mask[ii], 0);
      |               ^~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<float, unsigned char, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:5539:27: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
 5539 |       result_as_int[ii] = __byte_perm(src_reg, 0x4B000000, prmt_indices[ii]);
      |                           ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<cutlass::half_t, cutlass::integer_subbyte<2, true>, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:5678:20: error: there are no arguments to ‘__hfma2’ that depend on a template parameter, so a declaration of ‘__hfma2’ must be available [-fpermissive]
 5678 |       fp16x2_val = __hfma2(fp16x2_val,
      |                    ^~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<cutlass::half_t, cutlass::integer_subbyte<2, false>, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:5814:20: error: there are no arguments to ‘__hfma2’ that depend on a template parameter, so a declaration of ‘__hfma2’ must be available [-fpermissive]
 5814 |       fp16x2_val = __hfma2(fp16x2_val,
      |                    ^~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<cutlass::half_t, cutlass::integer_subbyte<4, true>, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:5960:20: error: there are no arguments to ‘__hfma2’ that depend on a template parameter, so a declaration of ‘__hfma2’ must be available [-fpermissive]
 5960 |       fp16x2_val = __hfma2(hfma_scale, fp16x2_val, hfma_bias);
      |                    ^~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static PackedResultType cutlass::NumericArrayConverter<cutlass::half_t, cutlass::integer_subbyte<4, false>, N, Round>::packed_convert(const PackedSrcType&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:6086:22: error: there are no arguments to ‘__hfma2’ that depend on a template parameter, so a declaration of ‘__hfma2’ must be available [-fpermissive]
 6086 |         fp16x2_val = __hfma2(fp16x2_val, reinterpret_cast<const __half2&>(hfma_scale), reinterpret_cast<const __half2&>(hfma_bias));
      |                      ^~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h: In static member function ‘static cutlass::FastNumericArrayConverter<signed char, float, 4, Round>::result_type cutlass::FastNumericArrayConverter<signed char, float, 4, Round>::convert(const source_type&)’:
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:7006:17: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
 7006 |     result[0] = __byte_perm(result[0], result[1], 0x40);
      |                 ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:7007:17: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
 7007 |     result[2] = __byte_perm(result[2], result[3], 0x40);
      |                 ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/numeric_conversion.h:7008:17: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
 7008 |     result[0] = __byte_perm(result[0], result[2], 0x5410);
      |                 ^~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_pipelined.h: In function ‘void cutlass::gemm::kernel::GemmPipelined(cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord, typename Mma::IteratorA::Params, typename Mma::IteratorA::TensorRef, typename Mma::IteratorB::Params, typename Mma::IteratorB::TensorRef, typename Epilogue::Params)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_pipelined.h:97:22: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread0’?
   97 |   int tb_thread_id = threadIdx.x;
      |                      ^~~~~~~~~
      |                      thread0
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h: At global scope:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:88:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
   88 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:262:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  262 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:359:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  359 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:458:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  458 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:654:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  654 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h:750:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  750 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:79:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
   79 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:253:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  253 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:374:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  374 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:498:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  498 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:654:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  654 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:762:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  762 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Crosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:871:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandRowMajorInterleaved<cutlass::sizeof_bits<Element>::value, InterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  871 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandRowMajorInterleaved<cutlass::sizeof_bits<Element>::value, InterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h:1021:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandColumnMajorInterleaved<cutlass::sizeof_bits<Element>::value, InterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1021 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandColumnMajorInterleaved<cutlass::sizeof_bits<Element>::value, InterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:141:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 64> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  141 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 64> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:539:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<32, 32> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  539 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<32, 32> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:907:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 32> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  907 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 32> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:1311:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 16> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1311 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, 16> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:1714:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1714 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:1951:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1951 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:2189:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 2189 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:2753:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 2753 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:2991:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 2991 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kCrosswise> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:3229:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 3229 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:3530:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 3530 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:3831:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 3831 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:4135:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 4135 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:55,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:4424:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<signed char, cutlass::layout::TensorNCxHWx<InterleavedN> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 4424 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<signed char, cutlass::layout::TensorNCxHWx<InterleavedN> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/layout/tensor_op_multiplicand_sm80.h: In member function ‘cutlass::layout::TensorOpMultiplicand64bCrosswise::LongIndex cutlass::layout::TensorOpMultiplicand64bCrosswise::operator()(const TensorCoord&) const’:
/content/STORM/cutlass/include/cutlass/layout/tensor_op_multiplicand_sm80.h:400:68: warning: suggest parentheses around arithmetic in operand of ‘^’ [-Wparentheses]
  400 |     int bank = ((k_group & 2) << 2) ^ ((s % 2) << 3) + (c % 4) * 2 + (access_s / 4) ^ (k_group & 1);
      |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/layout/tensor_op_multiplicand_sm80.h: In member function ‘cutlass::layout::TensorOpMultiplicandCrosswise128x4::LongIndex cutlass::layout::TensorOpMultiplicandCrosswise128x4::operator()(const TensorCoord&) const’:
/content/STORM/cutlass/include/cutlass/layout/tensor_op_multiplicand_sm80.h:943:22: warning: suggest parentheses around arithmetic in operand of ‘^’ [-Wparentheses]
  943 |     Index bank = liq + ((s & 1) * 4) ^ (c & 4);
      |                  ~~~~^~~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h: At global scope:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:122:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  122 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:418:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  418 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:644:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  644 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:877:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
  877 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:1193:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1193 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:1424:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1424 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:1656:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1656 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:2026:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 2026 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:38,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h:2253:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 2253 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:41,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h: In constructor ‘cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kA, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::FragmentShuffler()’:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:152:34: error: ‘LaneId’ is not a member of ‘cutlass::arch’
  152 |     int lane_id = cutlass::arch::LaneId();
      |                                  ^~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h: In member function ‘cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kA, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::WarpFragment cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kA, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::operator()(const WarpFragment&)’:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:174:25: error: there are no arguments to ‘__shfl_up_sync’ that depend on a template parameter, so a declaration of ‘__shfl_up_sync’ must be available [-fpermissive]
  174 |         uint32_t tmp0 = __shfl_up_sync(0xFFFFFFFF, src_ptr[0], delta_up_);
      |                         ^~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:175:25: error: there are no arguments to ‘__shfl_down_sync’ that depend on a template parameter, so a declaration of ‘__shfl_down_sync’ must be available [-fpermissive]
  175 |         uint32_t tmp1 = __shfl_down_sync(0xFFFFFFFF, src_ptr[0], delta_down_);
      |                         ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:176:25: error: there are no arguments to ‘__shfl_up_sync’ that depend on a template parameter, so a declaration of ‘__shfl_up_sync’ must be available [-fpermissive]
  176 |         uint32_t tmp2 = __shfl_up_sync(0xFFFFFFFF, src_ptr[1], delta_up_);
      |                         ^~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:177:25: error: there are no arguments to ‘__shfl_down_sync’ that depend on a template parameter, so a declaration of ‘__shfl_down_sync’ must be available [-fpermissive]
  177 |         uint32_t tmp3 = __shfl_down_sync(0xFFFFFFFF, src_ptr[1], delta_down_);
      |                         ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:180:22: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
  180 |         dst_ptr[0] = __byte_perm(tmp0, tmp2, byte_selector_);
      |                      ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:181:22: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
  181 |         dst_ptr[1] = __byte_perm(tmp1, tmp3, byte_selector_);
      |                      ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h: In constructor ‘cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kB, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::FragmentShuffler()’:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:236:34: error: ‘LaneId’ is not a member of ‘cutlass::arch’
  236 |     int lane_id = cutlass::arch::LaneId();
      |                                  ^~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h: In member function ‘cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kB, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::WarpFragment cutlass::gemm::warp::detail::FragmentShuffler<ElementMma_, ElementLoad_, NumMmaInstructions, NumElementsInWarpFragment, NumElementsInMmaFragment, cutlass::gemm::Operand::kB, typename std::enable_if<((cutlass::sizeof_bits<T>::value / cutlass::sizeof_bits<Element>::value) == 2)>::type>::operator()(const WarpFragment&)’:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:259:25: error: there are no arguments to ‘__shfl_up_sync’ that depend on a template parameter, so a declaration of ‘__shfl_up_sync’ must be available [-fpermissive]
  259 |         uint32_t tmp0 = __shfl_up_sync(0xFFFFFFFF, src_ptr[0], delta_up_);
      |                         ^~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:260:25: error: there are no arguments to ‘__shfl_down_sync’ that depend on a template parameter, so a declaration of ‘__shfl_down_sync’ must be available [-fpermissive]
  260 |         uint32_t tmp1 = __shfl_down_sync(0xFFFFFFFF, src_ptr[0], delta_down_);
      |                         ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:263:22: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
  263 |         dst_ptr[0] = __byte_perm(tmp0, tmp1, byte_selector_);
      |                      ^~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h: At global scope:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:122:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  122 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:399:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  399 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:624:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  624 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:841:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<cutlass::complex<A>, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  841 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<cutlass::complex<A>, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:1168:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1168 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:1453:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1453 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:1679:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1679 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:1915:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<cutlass::complex<float>, cutlass::layout::TensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1915 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<cutlass::complex<float>, cutlass::layout::TensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h:373,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h:121,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_complex_tensor_op_tile_iterator_sm80.h:2224:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<cutlass::complex<float>, cutlass::layout::TensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
 2224 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<cutlass::complex<float>, cutlass::layout::TensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h: In member function ‘void cutlass::gemm::threadblock::MmaPipelined<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, TransformA_, TransformB_, Enable>::gmem_wait()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:277:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  277 |     __syncthreads();
      |     ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_singlestage.h: In member function ‘void cutlass::gemm::threadblock::MmaSingleStage<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, Enable>::operator()(int, cutlass::gemm::threadblock::MmaSingleStage<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, Enable>::FragmentC&, cutlass::gemm::threadblock::MmaSingleStage<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, Enable>::IteratorA, cutlass::gemm::threadblock::MmaSingleStage<Shape_, IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, Enable>::IteratorB, const FragmentC&)’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_singlestage.h:217:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  217 |       __syncthreads();
      |       ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_singlestage.h:245:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  245 |       __syncthreads();
      |       ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h: At global scope:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:206:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_A>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  206 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_A>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:352:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  352 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:503:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  503 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:638:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_A>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  638 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, 2, Operator_>::Crosswise_A>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:765:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  765 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:904:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  904 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:1037:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1037 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<16, 64>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:1156:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1156 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16, 64>, cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, Shape_::kK>, float, LayoutC_, cutlass::arch::OpMultiplyAdd, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:1307:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, kInterleavedK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, kInterleavedK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK, AccumulatorsInRowMajor>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1307 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, kInterleavedK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, kInterleavedK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK, AccumulatorsInRowMajor>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:96:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
   96 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:312:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  312 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:442:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  442 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:569:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  569 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:785:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  785 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:915:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  915 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:1048:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kContiguous> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1048 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kContiguous> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:1265:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kRow> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1265 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kRow> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_tensor_op_sm70.h:1378:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kColumn> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1378 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<Element>::value, Shape_::kColumn> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:124:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  124 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:439:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  439 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:727:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  727 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:949:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  949 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<Element>::value> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:1168:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1168 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:1527:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1527 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::VoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:1859:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1859 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:2085:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 2085 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, kKBlock> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:2301:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
 2301 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_sm70.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:51,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm70.h:2652:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
 2652 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:209:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value>, ElementB_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementCompute_>::value>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  209 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value>, ElementB_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementCompute_>::value>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:362:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  362 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:517:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementCompute_>::value>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  517 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementAccumulator_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<cutlass::sizeof_bits<ElementCompute_>::value>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h:672:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value>, ElementB_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  672 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaVoltaTensorOp<WarpShape_, ElementA_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementAccumulator_>::value>, ElementB_, cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:117:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  117 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:327:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  327 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:580:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  580 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:790:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  790 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:1037:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1037 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:1249:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1249 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:1468:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<4> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1468 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<4> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:53,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_simt_tile_iterator.h:1694:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<4> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1694 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<4> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:61,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h:85:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
   85 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:61,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h:239:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  239 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:61,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h:336:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  336 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:85:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
   85 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:260:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  260 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:353:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  353 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:449:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
  449 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:635:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
  635 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:728:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise>’ changes meaning of ‘TensorRef’ [-fpermissive]
  728 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:824:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  824 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:999:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
  999 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:1092:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1092 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:1189:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1189 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::TensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:1368:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1368 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h:1461:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1461 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/warp/mma_gaussian_complex_tensor_op.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/warp/default_mma_complex_tensor_op.h:40,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/warp/mma_gaussian_complex_tensor_op_tile_iterator_sm80.h:119:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<cutlass::complex<A>, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  119 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<cutlass::complex<A>, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:61,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h: In member function ‘void cutlass::gemm::threadblock::MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_, Stages, SharedMemoryClear, Enable>::gmem_wait()’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h:490:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  490 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h: In member function ‘void cutlass::gemm::threadblock::MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_, Stages, SharedMemoryClear, Enable>::gemm_iters(int, cutlass::gemm::threadblock::MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_, Stages, SharedMemoryClear, Enable>::FragmentC&, cutlass::gemm::threadblock::MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_, Stages, SharedMemoryClear, Enable>::IteratorA&, cutlass::gemm::threadblock::MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_, SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_, Stages, SharedMemoryClear, Enable>::IteratorB&)’:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_multistage.h:666:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  666 |     __syncthreads();
      |     ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h: At global scope:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:197:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  197 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:327:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  327 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:460:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  460 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:592:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  592 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, InstructionShape_, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise128x4, cutlass::complex<double>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous128b, cutlass::complex<double>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:723:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  723 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:853:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  853 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:984:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  984 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:1115:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1115 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaComplexTensorOp<WarpShape_, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, cutlass::complex<float>, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, cutlass::complex<float>, LayoutC_, kTransformA, kTransformB, Operator_>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:1284:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1284 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:1451:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1451 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:1624:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1624 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::ColumnMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_multistage_mma_complex_core_sm80.h:1794:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1794 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, cutlass::complex<B>, cutlass::layout::ColumnMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMultistageMmaComplexCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, cutlass::complex<B>, cutlass::layout::RowMajor, cutlass::complex<C>, cutlass::layout::RowMajor, cutlass::complex<RealElementB>, LayoutC_, cutlass::arch::OpClassSimt, Stages, TransformA, TransformB, Operator_, CacheOpA, CacheOpB>::LaneN, 1> >, 1, kTransformA, kTransformB>, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:192:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, double, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  192 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, double, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:308:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, double, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  308 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous64b, double, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:422:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, double, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  422 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, double, cutlass::layout::ColumnMajorTensorOpMultiplicand64bCrosswise, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:540:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, double, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  540 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, double, cutlass::layout::RowMajorTensorOpMultiplicand64bCrosswise, double, cutlass::layout::RowMajorTensorOpMultiplicandCongruous64b, double, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1353:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_A>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1353 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_A>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1491:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1491 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1632:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_A>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1632 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementCompute_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_A>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, Shape_::kK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1772:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1772 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, Shape_::kK>, ElementB_, cutlass::layout::RowMajorTensorOpMultiplicandCongruous<cutlass::sizeof_bits<ElementC_>::value, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA, CacheOpB>::Crosswise_B>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1931:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, kInterleavedK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, kInterleavedK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK, AccumulatorsInRowMajor>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1931 |   using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<typename cutlass::gemm::warp::DefaultMmaTensorOp<WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementCompute_>::value, kInterleavedK>, ElementB_, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<cutlass::sizeof_bits<ElementC_>::value, kInterleavedK>, ElementC_, LayoutC_, Operator_, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK, AccumulatorsInRowMajor>::Type, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:2089:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 2089 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:2240:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 2240 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:2401:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 2401 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, (Shape_::kK / 32)>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:58,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h:2558:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 2558 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<4, 8>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, Stages, Operator_, false, CacheOpA, CacheOpB>::LaneN, 1> > >, cutlass::MatrixShape<(Shape_::kK / 32), 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), (Shape_::kK / WarpShape_::kK)>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:348:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
  348 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:349:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorView’ [-fpermissive]
  349 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:705:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  705 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:706:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  706 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:895:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  895 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:896:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  896 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1084:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<2> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1084 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<2> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1085:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<2> >’ changes meaning of ‘TensorView’ [-fpermissive]
 1085 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<2> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1392:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1392 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1393:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRank2ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
 1393 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRank2ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1575:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1575 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1576:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRank2RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
 1576 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRank2RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1762:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1762 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1763:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorView’ [-fpermissive]
 1763 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1953:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1953 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h:1954:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorView’ [-fpermissive]
 1954 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:173:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
  173 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:174:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorView’ [-fpermissive]
  174 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:432:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  432 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:433:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  433 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:650:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  650 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:651:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  651 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:860:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<2> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  860 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<2> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:861:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<2> >’ changes meaning of ‘TensorView’ [-fpermissive]
  861 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<2> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1111:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1111 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1112:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRank2ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
 1112 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRank2ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1319:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
 1319 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRank2RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1320:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRank2RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
 1320 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRank2RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1527:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1527 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1528:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorView’ [-fpermissive]
 1528 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajorInterleaved<kInterleavedK> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1720:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
 1720 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h:1721:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’ changes meaning of ‘TensorView’ [-fpermissive]
 1721 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajorInterleaved<kInterleavedK> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:98:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
   98 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:99:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorView’ [-fpermissive]
   99 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:497:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  497 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:498:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  498 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:676:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  676 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:43,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h:677:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  677 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:44,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h: In member function ‘void cutlass::transform::thread::Transpose<ElementCount_, cutlass::PitchLinearShape<4, 4>, signed char>::transform(cutlass::transform::thread::Transpose<ElementCount_, cutlass::PitchLinearShape<4, 4>, signed char>::Fragment&, cutlass::transform::thread::Transpose<ElementCount_, cutlass::PitchLinearShape<4, 4>, signed char>::Fragment&)’:
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:81:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   81 |       b0 = __byte_perm(a0, a1, 0x0040);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:82:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   82 |       c0 = __byte_perm(a2, a3, 0x0040);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:83:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   83 |       b0 = __byte_perm(b0, c0, 0x5410);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:85:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   85 |       b1 = __byte_perm(a0, a1, 0x0051);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:86:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   86 |       c0 = __byte_perm(a2, a3, 0x0051);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:87:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   87 |       b1 = __byte_perm(b1, c0, 0x5410);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:89:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   89 |       b2 = __byte_perm(a0, a1, 0x0062);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:90:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   90 |       c0 = __byte_perm(a2, a3, 0x0062);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:91:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   91 |       b2 = __byte_perm(b2, c0, 0x5410);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:93:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   93 |       b3 = __byte_perm(a0, a1, 0x0073);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:94:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   94 |       c0 = __byte_perm(a2, a3, 0x0073);
      |            ^~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/transform/thread/transpose.h:95:12: error: there are no arguments to ‘__byte_perm’ that depend on a template parameter, so a declaration of ‘__byte_perm’ must be available [-fpermissive]
   95 |       b3 = __byte_perm(b3, c0, 0x5410);
      |            ^~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h: At global scope:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:167:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
  167 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:168:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorView’ [-fpermissive]
  168 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::PitchLinear>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:420:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  420 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:421:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  421 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::ColumnMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:617:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  617 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h:618:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorView’ [-fpermissive]
  618 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::RowMajor>’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h:81:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
   81 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h:301:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  301 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:49,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h:436:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  436 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h:90:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’ changes meaning of ‘TensorRef’ [-fpermissive]
   90 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::PitchLinear>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h:281:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<4> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  281 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajorInterleaved<4> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h:403:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<4> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  403 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<4> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:230:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  230 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:399:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  399 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:564:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  564 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:729:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
  729 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::RowMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 1> > >, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_, cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:1231:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1231 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:1394:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1394 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:1553:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1553 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingM, 0>, cutlass::MatrixShape<0, 0>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::RowMajor, signed char, cutlass::layout::RowMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:59,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h:1713:9: error: declaration of ‘using MmaPolicy = struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’ changes meaning of ‘MmaPolicy’ [-fpermissive]
 1713 |   using MmaPolicy = MmaPolicy<
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_pipelined.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core.h:47,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/threadblock/mma_base.h:64:8: note: ‘MmaPolicy’ declared here as ‘struct cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaSimt<WarpShape_, signed char, cutlass::layout::ColumnMajorInterleaved<4>, signed char, cutlass::layout::RowMajorInterleaved<4>, ElementC_, LayoutC_, cutlass::gemm::warp::MmaSimtPolicy<cutlass::MatrixShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::WarpNumThreadsN>, cutlass::layout::ColumnMajorInterleaved<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneLayout>, cutlass::gemm::GemmShape<cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneM, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::LaneN, 4> >, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::kPaddingN>, cutlass::gemm::GemmShape<(Shape_::kM / WarpShape_::kM), (Shape_::kN / WarpShape_::kN), cutlass::gemm::threadblock::DefaultMmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, signed char, cutlass::layout::ColumnMajor, signed char, cutlass::layout::ColumnMajor, ElementC_, LayoutC_, cutlass::arch::OpClassSimt, 2, Operator_>::PartitionsK>::kK>’
   64 | struct MmaPolicy {
      |        ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h:78:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   78 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h:257:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedK> >’ changes meaning of ‘TensorRef’ [-fpermissive]
  257 |   using TensorRef = TensorRef<Element, TensorLayout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedK> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op.h:453:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
  453 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h:79:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   79 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h:326:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<int, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  326 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<int, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h:527:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<int, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  527 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<int, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h:716:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<float, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  716 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<float, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h:911:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<float, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  911 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<float, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:69,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h:95:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, typename std::conditional<(kRank == 4), cutlass::layout::TensorNHWC, cutlass::layout::TensorNDHWC>::type>’ changes meaning of ‘TensorRef’ [-fpermissive]
   95 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, typename std::conditional<(kRank == 4), cutlass::layout::TensorNHWC, cutlass::layout::TensorNDHWC>::type>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:70,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h:82:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   82 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:71,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h:86:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<Rank> >’ changes meaning of ‘TensorRef’ [-fpermissive]
   86 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::AffineRankN<Rank> >’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:71,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h:87:9: error: declaration of ‘using TensorView = class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<Rank> >’ changes meaning of ‘TensorView’ [-fpermissive]
   87 |   using TensorView = TensorView<Element, Layout>;
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_view.h:62:7: note: ‘TensorView’ declared here as ‘class cutlass::TensorView<Element_, cutlass::layout::AffineRankN<Rank> >’
   62 | class TensorView : public TensorRef<Element_, Layout_> {
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:72,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/shared_load_iterator.h:75:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   75 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:73,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/shared_load_iterator_mixed.h:97:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   97 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:73,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/shared_load_iterator_mixed.h:269:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  269 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:73,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/shared_load_iterator_mixed.h:438:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  438 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h:76,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:63,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/interleaved_epilogue.h: In member function ‘void cutlass::epilogue::threadblock::InterleavedEpilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, OutputOp_, InterleavedK>::unified(const OutputOp&, cutlass::epilogue::threadblock::InterleavedEpilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, OutputOp_, InterleavedK>::OutputTileIterator, const AccumulatorTile&, cutlass::epilogue::threadblock::InterleavedEpilogue<Shape_, WarpMmaOperator_, PartitionsK, OutputTileIterator_, AccumulatorFragmentIterator_, OutputOp_, InterleavedK>::OutputTileIterator)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/interleaved_epilogue.h:345:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  345 |       __syncthreads();  // Dummy (CUDA 11.0)
      |       ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h:64,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:64,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_volta_tensor_op.h: At global scope:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_volta_tensor_op.h:75:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   75 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h:64,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:64,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_volta_tensor_op.h:266:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<float, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  266 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<float, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_simt.h:80:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   80 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_simt.h:269:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  269 |   using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_simt.h:429:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
  429 |   using TensorRef = TensorRef<Element, Layout>;  ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/warp/tile_iterator_simt.h:554:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, Layout_>’ changes meaning of ‘TensorRef’ [-fpermissive]
  554 |   using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, Layout_>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:70,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/predicated_tile_iterator_direct_conv.h:87:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   87 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:72,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/shared_load_iterator_pitch_linear.h:75:9: error: declaration of ‘using TensorRef = class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’ changes meaning of ‘TensorRef’ [-fpermissive]
   75 |   using TensorRef = TensorRef<Element, Layout>;
      |         ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/tensor_view.h:50,
                 from /content/STORM/cutlass/include/cutlass/core_io.h:46,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h:48,
                 from /content/STORM/cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:42,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/tensor_ref.h:152:7: note: ‘TensorRef’ declared here as ‘class cutlass::TensorRef<Element_, cutlass::layout::RowMajor>’
  152 | class TensorRef {
      |       ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/epilogue/threadblock/default_epilogue_simt.h:74,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_depthwise.h: In member function ‘void cutlass::epilogue::threadblock::EpilogueDepthwise<Shape_, ThreadOutputShape_, ThreadBlockOutputShape_, WarpMmaOperator_, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_>::compute_source_needed_(const OutputOp&, cutlass::epilogue::threadblock::EpilogueDepthwise<Shape_, ThreadOutputShape_, ThreadBlockOutputShape_, WarpMmaOperator_, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_>::OutputTileIterator, const AccumulatorTile&, cutlass::epilogue::threadblock::EpilogueDepthwise<Shape_, ThreadOutputShape_, ThreadBlockOutputShape_, WarpMmaOperator_, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_>::OutputTileIterator)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_depthwise.h:240:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  240 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_depthwise.h: In member function ‘void cutlass::epilogue::threadblock::EpilogueDepthwise<Shape_, ThreadOutputShape_, ThreadBlockOutputShape_, WarpMmaOperator_, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_>::compute_source_not_needed_(const OutputOp&, cutlass::epilogue::threadblock::EpilogueDepthwise<Shape_, ThreadOutputShape_, ThreadBlockOutputShape_, WarpMmaOperator_, OutputTileIterator_, AccumulatorFragmentIterator_, WarpTileIterator_, SharedLoadIterator_, OutputOp_, Padding_>::OutputTileIterator, const AccumulatorTile&)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_depthwise.h:265:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  265 |     __syncthreads();
      |     ^~~~~~~~~~~~~
In file included from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm.h: In member function ‘cutlass::Status cutlass::gemm::device::Gemm<ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_, InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial, Operator_, GatherA, GatherB, ScatterD, PermuteDLayout>::run(cudaStream_t)’:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:495:34: error: expected primary-expression before ‘<’ token
  495 |     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
      |                                  ^
/content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:495:67: error: expected primary-expression before ‘>’ token
  495 |     cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
      |                                                                   ^
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:47,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/kernel_launch.h: In function ‘cutlass::Status cutlass::kernel_launch(dim3, dim3, size_t, cudaStream_t, const Params&, bool)’:
/content/STORM/cutlass/include/cutlass/kernel_launch.h:91:32: error: expected primary-expression before ‘<’ token
   91 |     device_kernel<GemmKernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(kernel_params);
      |                                ^
/content/STORM/cutlass/include/cutlass/kernel_launch.h:91:80: error: expected primary-expression before ‘>’ token
   91 |     device_kernel<GemmKernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(kernel_params);
      |                                                                                ^
In file included from /content/STORM/cutlass/include/cutlass/pipeline/pipeline.hpp:35,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp:40,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp:33,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp: In constructor ‘cutlass::PipelineTmaAsync<Stages_>::PipelineTmaAsync(cutlass::PipelineTmaAsync<Stages_>::SharedStorage&, cutlass::PipelineTmaAsync<Stages_>::Params, ClusterShape, InitBarriers, InitMasks)’:
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:334:22: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  334 |     int thread_idx = threadIdx.x;
      |                      ^~~~~~~~~
      |                      thread_idx
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp: In function ‘void cutlass::pipeline_init_wait(int)’:
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:1369:5: error: ‘__syncthreads’ was not declared in this scope
 1369 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp: In function ‘void cutlass::pipeline_init_arrive_relaxed(int)’:
/content/STORM/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:1382:5: error: ‘__syncthreads’ was not declared in this scope
 1382 |     __syncthreads();
      |     ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/pipeline/pipeline.hpp:36,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp:40,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp:33,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp: In member function ‘void cutlass::PipelineTmaTransformAsync<Stages_, AtomThrShape_MNK_>::init_masks(ClusterShape, dim3, cutlass::McastDirection)’:
/content/STORM/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp:399:24: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  399 |       int thread_idx = threadIdx.x;
      |                        ^~~~~~~~~
      |                        thread_idx
/content/STORM/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp: At global scope:
/content/STORM/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp:553:9: error: declaration of ‘using McastDirection = enum class cutlass::McastDirection’ changes meaning of ‘McastDirection’ [-fpermissive]
  553 |   using McastDirection = McastDirection;
      |         ^~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/pipeline/pipeline.hpp:36,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp:40,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp:33,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:62,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/pipeline/sm100_pipeline.hpp:48:12: note: ‘McastDirection’ declared here as ‘enum class cutlass::McastDirection’
   48 | enum class McastDirection {
      |            ^~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/barrier.h: In static member function ‘static void cutlass::detail::SyncthreadsSync::sync()’:
/content/STORM/cutlass/include/cutlass/barrier.h:53:5: error: ‘__syncthreads’ was not declared in this scope
   53 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/barrier.h: In static member function ‘static void cutlass::detail::SyncwarpSync::sync()’:
/content/STORM/cutlass/include/cutlass/barrier.h:60:5: error: ‘__syncwarp’ was not declared in this scope
   60 |     __syncwarp();
      |     ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/barrier.h: In static member function ‘static void cutlass::GenericBarrier<Sync>::red_release(int*, int)’:
/content/STORM/cutlass/include/cutlass/barrier.h:127:5: error: there are no arguments to ‘__threadfence’ that depend on a template parameter, so a declaration of ‘__threadfence’ must be available [-fpermissive]
  127 |     __threadfence();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/barrier.h:128:5: error: there are no arguments to ‘atomicAdd’ that depend on a template parameter, so a declaration of ‘atomicAdd’ must be available [-fpermissive]
  128 |     atomicAdd(ptr, val);
      |     ^~~~~~~~~
/content/STORM/cutlass/include/cutlass/barrier.h: In static member function ‘static void cutlass::GenericBarrier<Sync>::wait_eq_reset(void*, int, int, cutlass::GenericBarrier<Sync>::T)’:
/content/STORM/cutlass/include/cutlass/barrier.h:175:15: error: there are no arguments to ‘atomicCAS’ that depend on a template parameter, so a declaration of ‘atomicCAS’ must be available [-fpermissive]
  175 |         while(atomicCAS(flag_ptr, val, 0) != val) {}
      |               ^~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:65,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp: In constructor ‘cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::PersistentTileSchedulerSm90StreamK(const Params&)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:261:43: error: ‘blockIdx’ was not declared in this scope
  261 |       current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
      |                                           ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:261:89: error: ‘gridDim’ was not declared in this scope
  261 |       current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
      |                                                                                         ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:264:43: error: ‘blockIdx’ was not declared in this scope
  264 |       current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
      |                                           ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:264:66: error: ‘gridDim’ was not declared in this scope
  264 |       current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
      |                                                                  ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp: In member function ‘void cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::advance_to_next_work(uint32_t)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:326:42: error: ‘gridDim’ was not declared in this scope
  326 |     current_work_linear_idx_ += uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z) * uint64_t(advance_count);
      |                                          ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp: In member function ‘bool cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::is_last_tile(cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::WorkTileInfo, uint32_t) const’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:339:20: error: ‘gridDim’ was not declared in this scope
  339 |           uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z) * uint64_t(advance_count)
      |                    ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp: In static member function ‘static void cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::fixup_helper(const Params&, const cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>::WorkTileInfo&, FrgTensorC&, uint32_t, uint32_t, uint32_t, uint32_t)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp:472:41: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread’?
  472 |     uint32_t barrier_group_thread_idx = threadIdx.x % BarrierManager::ThreadCount;
      |                                         ^~~~~~~~~
      |                                         thread
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:66,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp: In static member function ‘static WorkTileInfo cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>::get_work_idx_m_and_n(uint64_t, GroupInfo&, GroupProblemShape&, ProblemShape (&)[2], cutlass::gemm::GemmCoord, cutlass::gemm::GemmCoord, const cutlass::FastDivmodU64Pow2&, const cutlass::FastDivmodU64Pow2&, const cutlass::FastDivmodU64&, const cutlass::FastDivmodU64&, int32_t, RasterOrder)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp:349:41: error: there are no arguments to ‘__ffs’ that depend on a template parameter, so a declaration of ‘__ffs’ must be available [-fpermissive]
  349 |           int first_succeeding_thread = __ffs(thread_succeed) - 1;
      |                                         ^~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp:378:30: error: ‘blockIdx’ was not declared in this scope
  378 |       cluster_minor_offset = blockIdx.x;
      |                              ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp:381:30: error: ‘blockIdx’ was not declared in this scope
  381 |       cluster_minor_offset = blockIdx.y;
      |                              ^~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:67,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp: In member function ‘cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape_, Stages_>::WorkTileInfo cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape_, Stages_>::initial_work_tile_info(ClusterShape)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:369:34: error: ‘blockIdx’ was not declared in this scope
  369 |     return swizzle_and_rasterize(blockIdx.x, blockIdx.y, blockIdx.z, /*valid=*/true, /*cluster_offset_m=*/0, /*cluster_offset_n=*/0);
      |                                  ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp: In member function ‘cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape_, Stages_>::WorkTileInfo cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape_, Stages_>::swizzle_and_rasterize(int, int, int, bool, int, int) const’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp:772:71: error: ‘gridDim’ was not declared in this scope
  772 |       int32_t major_clusters = params_.divmod_cluster_shape_m_.divide(gridDim.x);
      |                                                                       ^~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/tile_scheduler.hpp:68,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:34,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp: In member function ‘uint64_t cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100StreamK<TileShape, ClusterShape, Stages_>::to_linear_idx(const InternalWorkTileInfo&, const Params&)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp:724:28: error: ‘gridDim’ was not declared in this scope
  724 |     uint64_t cluster_idx = gridDim.y * start_cta_m_preferred_cluster + start_cta_n_preferred_cluster;
      |                            ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp: In member function ‘AccumulatorPipelineState cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100StreamK<TileShape, ClusterShape, Stages_>::tmem_fixup(const TiledMma&, const WorkTileInfo&, cute::Tensor<AccEngine, AccLayout>&, AccumulatorPipeline, AccumulatorPipelineState, CopyOpT2R, uint32_t, uint32_t) const’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp:916:46: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread’?
  916 |     auto thr_tmem_load = tmem_load.get_slice(threadIdx.x % ThreadsForFixup);
      |                                              ^~~~~~~~~
      |                                              thread
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:56,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_base_of_v<cutlass::gemm::KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_base_of_v<cutlass::gemm::KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm.hpp:210:26: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  210 |     int thread_idx = int(threadIdx.x);
      |                          ^~~~~~~~~
      |                          thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm.hpp:212:59: error: ‘blockIdx’ was not declared in this scope
  212 |     auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
      |                                                           ^~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:57,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm_array.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_base_of_v<cutlass::gemm::KernelPtrArrayMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_base_of_v<cutlass::gemm::KernelPtrArrayMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm_array.hpp:217:26: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  217 |     int thread_idx = int(threadIdx.x);
      |                          ^~~~~~~~~
      |                          thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm70_gemm_array.hpp:219:59: error: ‘blockIdx’ was not declared in this scope
  219 |     auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
      |                                                           ^~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:67,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedSm100>, cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100> >, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedSm100>, cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm100> >, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:663:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  663 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp:728:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  728 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:68,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedMmaTransformSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedMmaTransformSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp:693:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  693 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp:745:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  745 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mma_transform.hpp:826:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  826 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:69,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedSm100>, cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100> >, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<disjunction_v<cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedSm100>, cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm100> >, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:679:57: error: ‘SmId’ is not a member of ‘cutlass::arch’
  679 |     int32_t sm_id = static_cast<int32_t>(cutlass::arch::SmId());
      |                                                         ^~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:702:15: error: ‘blockIdx’ was not declared in this scope
  702 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |               ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:702:42: error: ‘gridDim’ was not declared in this scope
  702 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |                                          ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:790:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  790 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp:872:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  872 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:70,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_input_transform.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedInputTransformSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedInputTransformSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_input_transform.hpp:412:35: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  412 |     int thread_idx          = int(threadIdx.x);
      |                                   ^~~~~~~~~
      |                                   thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_input_transform.hpp:695:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  695 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_input_transform.hpp:818:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  818 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:71,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedMixedInputTransformSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedMixedInputTransformSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp:414:35: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  414 |     int thread_idx          = int(threadIdx.x);
      |                                   ^~~~~~~~~
      |                                   thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp:753:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  753 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp:878:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  878 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:72,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedInputTransformSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedInputTransformSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:430:35: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  430 |     int thread_idx          = int(threadIdx.x);
      |                                   ^~~~~~~~~
      |                                   thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:686:72: error: ‘SmId’ is not a member of ‘cutlass::arch’
  686 |           params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId()));
      |                                                                        ^~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:712:11: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  712 |           __syncwarp();
      |           ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:767:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  767 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:902:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  902 |       __syncwarp();
      |       ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:976:125: error: ‘SmId’ is not a member of ‘cutlass::arch’
  976 |           params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId())));
      |                                                                                                                             ^~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:999:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  999 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp:1067:125: error: ‘SmId’ is not a member of ‘cutlass::arch’
 1067 |           params.epilogue, shared_storage.tensormaps.epilogue, params.hw_info.sm_count, static_cast<int32_t>(cutlass::arch::SmId())));
      |                                                                                                                             ^~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:73,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedMmaTransformSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedMmaTransformSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:723:57: error: ‘SmId’ is not a member of ‘cutlass::arch’
  723 |     int32_t sm_id = static_cast<int32_t>(cutlass::arch::SmId());
      |                                                         ^~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:751:15: error: ‘blockIdx’ was not declared in this scope
  751 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |               ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:751:42: error: ‘gridDim’ was not declared in this scope
  751 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |                                          ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:844:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  844 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:912:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  912 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp:1004:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
 1004 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:74,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<(is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelSparseTmaWarpSpecializedSm100> || is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100>), void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<(is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelSparseTmaWarpSpecializedSm100> || is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelSparseTmaWarpSpecializedBlockScaledSm100>), void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp:703:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  703 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp:768:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  768 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:75,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelWarpSpecializedSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelWarpSpecializedSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp:562:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  562 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp:621:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  621 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:76,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecializedSm100>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelMixedTmaCpAsyncWarpSpecializedSm100>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp:705:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  705 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp:751:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  751 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp:826:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  826 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:77,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm103>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedBlockScaledSm103>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp:720:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  720 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp:847:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  847 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp:869:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  869 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:78,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm103>, void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockScaledSm103>, void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:732:57: error: ‘SmId’ is not a member of ‘cutlass::arch’
  732 |     int32_t sm_id = static_cast<int32_t>(cutlass::arch::SmId());
      |                                                         ^~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:761:15: error: ‘blockIdx’ was not declared in this scope
  761 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |               ^~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:761:42: error: ‘gridDim’ was not declared in this scope
  761 |       sm_id = blockIdx.x + (blockIdx.y * gridDim.x);
      |                                          ^~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:847:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
  847 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:1002:9: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
 1002 |         __syncwarp();
      |         ^~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp:1028:7: error: there are no arguments to ‘__syncwarp’ that depend on a template parameter, so a declaration of ‘__syncwarp’ must be available [-fpermissive]
 1028 |       __syncwarp();
      |       ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.hpp:79,
                 from /content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:45,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp: In member function ‘void cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<(is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseSm120> || is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120>), void>::type>::operator()(const cutlass::gemm::kernel::GemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_, typename std::enable_if<(is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseSm120> || is_asymmetric_dma_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, cutlass::gemm::KernelTmaWarpSpecializedCooperativeSparseBlockScaledSm120>), void>::type>::Params&, char*)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp:374:26: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  374 |     int thread_idx = int(threadIdx.x);
      |                          ^~~~~~~~~
      |                          thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp: In lambda function:
/content/STORM/cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp:512:9: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  512 |         __syncthreads();
      |         ^~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:50,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h: In member function ‘void cutlass::gemm::kernel::GemmUniversal<Mma_, Epilogue_, ThreadblockSwizzle_, void, typename std::enable_if<(!(cute::is_tuple<T>::value || cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape_>::value)), void>::type>::run_with_swizzle(const cutlass::gemm::kernel::GemmUniversal<Mma_, Epilogue_, ThreadblockSwizzle_, void, typename std::enable_if<(!(cute::is_tuple<T>::value || cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape_>::value)), void>::type>::Params&, cutlass::gemm::kernel::GemmUniversal<Mma_, Epilogue_, ThreadblockSwizzle_, void, typename std::enable_if<(!(cute::is_tuple<T>::value || cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape_>::value)), void>::type>::SharedStorage&, cutlass::gemm::kernel::GemmUniversal<Mma_, Epilogue_, ThreadblockSwizzle_, void, typename std::enable_if<(!(cute::is_tuple<T>::value || cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape_>::value)), void>::type>::ThreadblockSwizzle&)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:514:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  514 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal.h:528:22: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  528 |     int thread_idx = threadIdx.x;
      |                      ^~~~~~~~~
      |                      thread_idx
In file included from /content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm_universal.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:52,
                 from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h: In member function ‘typename cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::Mma::IteratorA cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::init_iterator_A(cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::TileWorkDesc&, cutlass::gemm::GemmUniversalMode)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:661:9: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  661 |         threadIdx.x,
      |         ^~~~~~~~~
      |         thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h: In member function ‘typename cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::Mma::IteratorB cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::init_iterator_B(cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::TileWorkDesc&, cutlass::gemm::GemmUniversalMode)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:690:9: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
  690 |         threadIdx.x,
      |         ^~~~~~~~~
      |         thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h: In member function ‘void cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::gemm()’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:1095:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
 1095 |       __syncthreads();
      |       ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h: In constructor ‘cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::GemmUniversalStreamk(const cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::Params&, cutlass::gemm::kernel::GemmUniversalStreamk<Mma_, Epilogue_, ThreadblockSwizzle_>::SharedStorage&)’:
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:1139:18: error: ‘threadIdx’ was not declared in this scope; did you mean ‘thread_idx’?
 1139 |       thread_idx(threadIdx.x),
      |                  ^~~~~~~~~
      |                  thread_idx
/content/STORM/cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h:1140:16: error: there are no arguments to ‘__shfl_sync’ that depend on a template parameter, so a declaration of ‘__shfl_sync’ must be available [-fpermissive]
 1140 |       warp_idx(__shfl_sync(0xffffffff, threadIdx.x / 32, 0)),   // broadcast the warp_id computed by lane 0 to ensure dependent code
      |                ^~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h: In member function ‘cutlass::Status cutlass::gemm::device::GemmUniversalBase<GemmKernel_>::run(cudaStream_t, cutlass::CudaHostAdapter*)’:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:460:28: error: expected primary-expression before ‘<’ token
  460 |       Kernel2<GemmKernel><<<grid, block, kSharedStorageSize, stream>>>(params_);
      |                            ^
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:460:70: error: expected primary-expression before ‘>’ token
  460 |       Kernel2<GemmKernel><<<grid, block, kSharedStorageSize, stream>>>(params_);
      |                                                                      ^
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:57,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h: At global scope:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:124:11: error: declaration of ‘using ElementAccumulator = using ElementAccumulator = typename DefaultEpilogue::WarpTileIterator::Element’ changes meaning of ‘ElementAccumulator’ [-fpermissive]
  124 |     using ElementAccumulator = ElementAccumulator;
      |           ^~~~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:121:9: note: ‘ElementAccumulator’ declared here as ‘using ElementAccumulator = typename DefaultEpilogue::WarpTileIterator::Element’
  121 |   using ElementAccumulator = typename WarpTileIterator::Element;
      |         ^~~~~~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:57,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h: In member function ‘void cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<DefaultEpilogue, FusionCallbacks_, Stages, IterationsUnroll>::reduce(int, int, int, void*, cutlass::gemm::GemmCoord, ProblemShape, int)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:201:5: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  201 |     __syncthreads();
      |     ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h: In member function ‘void cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<DefaultEpilogue, FusionCallbacks_, Stages, IterationsUnroll>::operator()(const AccumulatorTile&, cutlass::gemm::GemmCoord, ProblemShape, int)’:
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:287:7: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  287 |       __syncthreads();
      |       ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:315:9: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  315 |         __syncthreads();
      |         ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:412:9: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  412 |         __syncthreads();
      |         ^~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h:417:9: error: there are no arguments to ‘__syncthreads’ that depend on a template parameter, so a declaration of ‘__syncthreads’ must be available [-fpermissive]
  417 |         __syncthreads();
      |         ^~~~~~~~~~~~~
In file included from storm_bindings.cpp:13:
storm_gemm.h: In static member function ‘static cudaError_t storm::StormCUTLASSGEMM<Element, LayoutA, LayoutB, LayoutC>::execute(const Element*, const Element*, Element*, int, int, int, Element, Element, cudaStream_t)’:
storm_gemm.h:161:20: error: ‘cudaErrorLaunchFailed’ was not declared in this scope; did you mean ‘cudaErrorLaunchFailure’?
  161 |             return cudaErrorLaunchFailed;
      |                    ^~~~~~~~~~~~~~~~~~~~~
      |                    cudaErrorLaunchFailure
storm_gemm.h: In static member function ‘static cudaError_t storm::StormGEMM::storm_gemm_with_bias(const float*, const float*, const float*, float*, int, int, int, cudaStream_t)’:
storm_gemm.h:244:26: error: expected primary-expression before ‘<’ token
  244 |         add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
      |                          ^
storm_gemm.h:244:51: error: expected primary-expression before ‘>’ token
  244 |         add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
      |                                                   ^
storm_gemm.h:244:53: warning: left operand of comma operator has no effect [-Wunused-value]
  244 |         add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
      |                                                     ^
storm_gemm.h:244:62: warning: right operand of comma operator has no effect [-Wunused-value]
  244 |         add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
      |                                                              ^
storm_gemm.h:244:65: warning: right operand of comma operator has no effect [-Wunused-value]
  244 |         add_bias_kernel<<<grid, block, 0, stream>>>(C, bias, M, N);
      |                                                                 ^
storm_gemm.h: In static member function ‘static void storm::StormGEMM::add_bias_kernel(float*, const float*, int, int)’:
storm_gemm.h:261:19: error: ‘blockIdx’ was not declared in this scope
  261 |         int col = blockIdx.x * blockDim.x + threadIdx.x;
      |                   ^~~~~~~~
storm_gemm.h:261:32: error: ‘blockDim’ was not declared in this scope
  261 |         int col = blockIdx.x * blockDim.x + threadIdx.x;
      |                                ^~~~~~~~
storm_gemm.h:261:45: error: ‘threadIdx’ was not declared in this scope
  261 |         int col = blockIdx.x * blockDim.x + threadIdx.x;
      |                                             ^~~~~~~~~
In file included from /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/Exceptions.h:12,
                 from /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include/torch/python.h:11,
                 from /usr/local/lib/python3.12/dist-packages/torch/include/torch/extension.h:9,
                 from storm_bindings.cpp:7:
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<StormGEMMTensor>’:
storm_bindings.cpp:119:64:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<StormGEMMTensor>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
 1539 | class class_ : public detail::generic_type {
      |       ^~~~~~
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<storm::CUDAStream>’:
storm_bindings.cpp:124:61:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<storm::CUDAStream>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<storm::LayerEventManager>’:
storm_bindings.cpp:129:75:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<storm::LayerEventManager>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘class pybind11::class_<StormModel>’:
storm_bindings.cpp:136:54:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1539:7: warning: ‘pybind11::class_<StormModel>’ declared with greater visibility than its base ‘pybind11::detail::generic_type’ [-Wattributes]
In file included from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm.h: In instantiation of ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:620:97:   recursively required by substitution of ‘template<class GemmKernel> struct cutlass::gemm::detail::IsCutlass3GemmKernel<GemmKernel, std::void_t<typename GemmKernel::ProblemShape> > [with GemmKernel = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>]’
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:620:97:   required by substitution of ‘template<class GemmKernel_> class cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_, typename std::enable_if<(! cutlass::gemm::detail::IsCutlass3GemmKernel<typename cutlass::GetUnderlyingKernel<T>::type>::value), void>::type> [with GemmKernel_ = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>]’
storm_gemm.h:115:43:   required from ‘static cudaError_t storm::StormCUTLASSGEMM<Element, LayoutA, LayoutB, LayoutC>::execute(const Element*, const Element*, Element*, int, int, int, Element, Element, cudaStream_t) [with Element = float; LayoutA = cutlass::layout::RowMajor; LayoutB = cutlass::layout::ColumnMajor; LayoutC = cutlass::layout::RowMajor; cudaError_t = cudaError; cudaStream_t = CUstream_st*]’
storm_gemm.h:203:60:   required from here
/content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:264:9: error: invalid use of incomplete type ‘struct cutlass::gemm::kernel::DefaultGemm<float, cutlass::layout::RowMajor, 1, float, cutlass::layout::ColumnMajor, 1, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, false, cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, cutlass::layout::NoPermute, void>’
  264 |   using GemmKernel = typename kernel::DefaultGemm<
      |         ^~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm.h:45,
                 from storm_gemm.h:14,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/kernel/default_gemm.h:139:8: note: declaration of ‘struct cutlass::gemm::kernel::DefaultGemm<float, cutlass::layout::RowMajor, 1, float, cutlass::layout::ColumnMajor, 1, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, false, cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, cutlass::layout::NoPermute, void>’
  139 | struct DefaultGemm;
      |        ^~~~~~~~~~~
In file included from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h: In instantiation of ‘class cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>, void>’:
storm_gemm.h:115:43:   required from ‘static cudaError_t storm::StormCUTLASSGEMM<Element, LayoutA, LayoutB, LayoutC>::execute(const Element*, const Element*, Element*, int, int, int, Element, Element, cudaStream_t) [with Element = float; LayoutA = cutlass::layout::RowMajor; LayoutB = cutlass::layout::ColumnMajor; LayoutC = cutlass::layout::RowMajor; cudaError_t = cudaError; cudaStream_t = CUstream_st*]’
storm_gemm.h:203:60:   required from here
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:630:9: error: no type named ‘Mma’ in ‘using GemmKernel = cutlass::GetUnderlyingKernel_t<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute> >’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
  630 |   using ThreadblockShape = typename GemmKernel::Mma::Shape;
      |         ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:635:9: error: no type named ‘Mma’ in ‘using GemmKernel = cutlass::GetUnderlyingKernel_t<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute> >’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
  635 |   using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
      |         ^~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h: In instantiation of ‘const bool cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>, void>::kInternalTranspose’:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:645:9:   required from ‘class cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>, void>’
storm_gemm.h:115:43:   required from ‘static cudaError_t storm::StormCUTLASSGEMM<Element, LayoutA, LayoutB, LayoutC>::execute(const Element*, const Element*, Element*, int, int, int, Element, Element, cudaStream_t) [with Element = float; LayoutA = cutlass::layout::RowMajor; LayoutB = cutlass::layout::ColumnMajor; LayoutC = cutlass::layout::RowMajor; cudaError_t = cudaError; cudaStream_t = CUstream_st*]’
storm_gemm.h:203:60:   required from here
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:627:46: error: no type named ‘Epilogue’ in ‘using GemmKernel = cutlass::GetUnderlyingKernel_t<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute> >’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
  627 |     !cutlass::epilogue::threadblock::detail::is_2x_evt_v<typename GemmKernel::Epilogue> &&  // 2.x EVT does not require internal transpose
      |      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h: In instantiation of ‘class cutlass::gemm::device::GemmUniversalBase<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute> >’:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:687:9:   required from ‘class cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>, void>’
storm_gemm.h:115:43:   required from ‘static cudaError_t storm::StormCUTLASSGEMM<Element, LayoutA, LayoutB, LayoutC>::execute(const Element*, const Element*, Element*, int, int, int, Element, Element, cudaStream_t) [with Element = float; LayoutA = cutlass::layout::RowMajor; LayoutB = cutlass::layout::ColumnMajor; LayoutC = cutlass::layout::RowMajor; cudaError_t = cudaError; cudaStream_t = CUstream_st*]’
storm_gemm.h:203:60:   required from here
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:75:9: error: no type named ‘Mma’ in ‘using GemmKernel = class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
   75 |   using ThreadblockShape = typename GemmKernel::Mma::Shape;
      |         ^~~~~~~~~~~~~~~~
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:93:9: error: no type named ‘Mma’ in ‘using GemmKernel = class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
   93 |   using ElementAccumulator = typename GemmKernel::Mma::ElementC;
      |         ^~~~~~~~~~~~~~~~~~
In file included from /content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:54,
                 from storm_gemm.h:15,
                 from storm_bindings.cpp:13:
/content/STORM/cutlass/include/cutlass/gemm/device/gemm_universal_base.h:202:31: error: no type named ‘Params’ in ‘using GemmKernel = class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’ {aka ‘class cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75, cutlass::gemm::GemmShape<64, 64, 8>, cutlass::gemm::GemmShape<8, 8, 8>, cutlass::gemm::GemmShape<4, 4, 2>, cutlass::epilogue::thread::LinearCombination<float, 1, float, float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, float>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, 1, 1, false, cutlass::arch::OpMultiplyAdd, false, false, false, cutlass::layout::NoPermute>’}
  202 |   typename GemmKernel::Params params_;
      |                               ^~~~~~~
In file included from /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/Exceptions.h:12,
                 from /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include/torch/python.h:11,
                 from /usr/local/lib/python3.12/dist-packages/torch/include/torch/extension.h:9,
                 from storm_bindings.cpp:7:
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = StormGEMMTensor; options = {}]’:
storm_bindings.cpp:119:64:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<StormGEMMTensor>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<StormGEMMTensor>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
 1599 |             with_internals([&](internals &internals) {
      |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
 1600 |                 auto &instances = record.module_local ? get_local_internals().registered_types_cpp
      |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 1601 |                                                       : internals.registered_types_cpp;
      |                                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 1602 |                 instances[std::type_index(typeid(type_alias))]
      |                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 1603 |                     = instances[std::type_index(typeid(type))];
      |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 1604 |             });
      |             ~               
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = storm::CUDAStream; options = {}]’:
storm_bindings.cpp:124:61:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<storm::CUDAStream>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<storm::CUDAStream>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = storm::LayerEventManager; options = {}]’:
storm_bindings.cpp:129:75:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<storm::LayerEventManager>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<storm::LayerEventManager>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h: In instantiation of ‘pybind11::class_< <template-parameter-1-1>, <template-parameter-1-2> >::class_(pybind11::handle, const char*, const Extra& ...) [with Extra = {}; type_ = StormModel; options = {}]’:
storm_bindings.cpp:136:54:   required from here
/usr/local/lib/python3.12/dist-packages/torch/include/pybind11/pybind11.h:1599:28: warning: ‘pybind11::class_<StormModel>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>’ declared with greater visibility than the type of its field ‘pybind11::class_<StormModel>::class_<>(pybind11::handle, const char*)::<lambda(pybind11::detail::internals&)>::<record capture>’ [-Wattributes]
error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1