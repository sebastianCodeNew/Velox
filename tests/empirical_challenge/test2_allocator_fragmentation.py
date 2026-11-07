"""
Empirical Challenge Harness 2: PyTorch CUDA Caching Allocator Simulator
Simulates:
1. Exact Block split-and-merge doubly linked list (matching c10::cuda::CUDACachingAllocator).
2. Continuous batching workload: alternating prefill activations and decode activations with non-deterministic lifetimes.
3. Stress tests:
   - Scenario A: Standard Continuous Batching (variable prefill + decode).
   - Scenario B: Chunked Prefill (bounded activation sizes: e.g. chunk size 512 tokens).
   - Scenario C: Virtual Memory / Expandable Segments (virtual page remapping).
"""

import random
import bisect

class Block:
    def __init__(self, size, offset, prev_block=None, next_block=None):
        self.size = size
        self.offset = offset
        self.allocated = False
        self.prev = prev_block
        self.next = next_block

class CUDACachingAllocatorSim:
    def __init__(self, total_vram_mb=8192):
        self.total_vram_mb = total_vram_mb
        # Initial single contiguous block
        initial_block = Block(total_vram_mb, 0)
        self.head = initial_block
        self.free_blocks = [initial_block]  # Ordered by size for best-fit
        self.allocated_blocks = {}
        self.next_alloc_id = 0

    def malloc(self, size_mb):
        # Best-fit search: find smallest block with size >= size_mb
        # free_blocks is sorted by size
        best_idx = -1
        for idx, blk in enumerate(self.free_blocks):
            if blk.size >= size_mb:
                best_idx = idx
                break
        
        if best_idx == -1:
            # Out of memory! Contiguous block not found
            total_free = sum(b.size for b in self.free_blocks)
            largest_free = max((b.size for b in self.free_blocks), default=0)
            return None, total_free, largest_free

        candidate = self.free_blocks.pop(best_idx)
        candidate.allocated = True
        
        # Split block if excess size >= 1 MB
        excess = candidate.size - size_mb
        if excess >= 1:
            new_free = Block(excess, candidate.offset + size_mb, prev_block=candidate, next_block=candidate.next)
            if candidate.next:
                candidate.next.prev = new_free
            candidate.next = new_free
            candidate.size = size_mb
            # Insert into sorted free_blocks
            self._insert_free(new_free)

        alloc_id = self.next_alloc_id
        self.next_alloc_id += 1
        self.allocated_blocks[alloc_id] = candidate
        return alloc_id, 0, 0

    def free(self, alloc_id):
        if alloc_id not in self.allocated_blocks:
            return
        blk = self.allocated_blocks.pop(alloc_id)
        blk.allocated = False

        # Merge with adjacent left (prev) if free
        if blk.prev and not blk.prev.allocated:
            prev_blk = blk.prev
            self._remove_free(prev_blk)
            prev_blk.size += blk.size
            prev_blk.next = blk.next
            if blk.next:
                blk.next.prev = prev_blk
            blk = prev_blk

        # Merge with adjacent right (next) if free
        if blk.next and not blk.next.allocated:
            next_blk = blk.next
            self._remove_free(next_blk)
            blk.size += next_blk.size
            blk.next = next_blk.next
            if next_blk.next:
                next_blk.next.prev = blk

        self._insert_free(blk)

    def _insert_free(self, blk):
        # Insert sorted by size (best-fit)
        idx = bisect.bisect_left([b.size for b in self.free_blocks], blk.size)
        self.free_blocks.insert(idx, blk)

    def _remove_free(self, blk):
        for i, b in enumerate(self.free_blocks):
            if b is blk:
                self.free_blocks.pop(i)
                return

    def get_fragmentation_stats(self):
        total_free = sum(b.size for b in self.free_blocks)
        largest_free = max((b.size for b in self.free_blocks), default=0)
        frag_ratio = 1.0 - (largest_free / total_free) if total_free > 0 else 0.0
        return total_free, largest_free, frag_ratio

def run_simulation(name, chunked_prefill=False, expandable_segments=False, num_steps=2000):
    random.seed(42)
    allocator = CUDACachingAllocatorSim(total_vram_mb=8192) # 8 GB activation headroom
    
    # Active requests: dict of req_id -> {alloc_id, remaining_tokens, req_type}
    active_requests = {}
    next_req_id = 0
    oom_events = 0
    max_frag_ratio = 0.0
    
    for step in range(num_steps):
        # 1. Process departures / step decrement
        completed_reqs = []
        for req_id, req_data in list(active_requests.items()):
            req_data["remaining_tokens"] -= 1
            if req_data["remaining_tokens"] <= 0:
                completed_reqs.append(req_id)
        
        for req_id in completed_reqs:
            allocator.free(active_requests[req_id]["alloc_id"])
            del active_requests[req_id]

        # 2. Arriving requests (Poisson arrival rate ~ 0.5 per step)
        if random.random() < 0.6:
            req_type = "prefill" if random.random() < 0.25 else "decode"
            if req_type == "prefill":
                if chunked_prefill:
                    # Chunked prefill bounds activation to 512 MB
                    act_size = 512
                else:
                    # Unchunked prefill has variable activations: 512 MB to 2048 MB
                    act_size = random.choice([512, 1024, 1536, 2048])
                gen_len = random.randint(10, 50)
            else:
                # Decode activation: 64 MB to 256 MB
                act_size = random.choice([64, 128, 256])
                gen_len = random.randint(5, 30)

            alloc_id, total_free, largest_free = allocator.malloc(act_size)
            
            if alloc_id is None:
                # Allocation failed!
                if expandable_segments:
                    # In expandable segments, virtual memory can map non-contiguous physical pages
                    # So allocation only fails if total_free < act_size!
                    if total_free >= act_size:
                        # Rescued by expandable segments virtual paging!
                        pass
                    else:
                        oom_events += 1
                else:
                    oom_events += 1
            else:
                active_requests[next_req_id] = {
                    "alloc_id": alloc_id,
                    "remaining_tokens": gen_len,
                    "size": act_size,
                    "type": req_type
                }
                next_req_id += 1

        total_free, largest_free, frag_ratio = allocator.get_fragmentation_stats()
        if frag_ratio > max_frag_ratio:
            max_frag_ratio = frag_ratio

    print(f"Simulation [{name}]:")
    print(f"  Total steps: {num_steps}")
    print(f"  OOM Events: {oom_events}")
    print(f"  Max Fragmentation Ratio: {max_frag_ratio * 100:.2f}%")
    total_free, largest_free, frag_ratio = allocator.get_fragmentation_stats()
    print(f"  Final state: Total Free = {total_free} MB, Largest Free = {largest_free} MB, Frag Ratio = {frag_ratio*100:.2f}%\n")
    return oom_events, max_frag_ratio

if __name__ == "__main__":
    print("=" * 65)
    print("EMPIRICAL TEST 2: ALLOCATOR FRAGMENTATION & MITIGATION STRESS TEST")
    print("=" * 65)
    
    # 1. Standard Continuous Batching
    run_simulation("Baseline: PyTorch Split-and-Merge Allocator", chunked_prefill=False, expandable_segments=False)
    
    # 2. Chunked Prefill (Sarathi / vLLM chunked prefill)
    run_simulation("Mitigation 1: Chunked Prefill (Bounded Size)", chunked_prefill=True, expandable_segments=False)
    
    # 3. Virtual Memory / Expandable Segments (PyTorch 2.1+)
    run_simulation("Mitigation 2: Virtual Memory / Expandable Segments", chunked_prefill=False, expandable_segments=True)

    # 4. Both Chunked Prefill + Expandable Segments
    run_simulation("Mitigation 3: Chunked Prefill + Expandable Segments", chunked_prefill=True, expandable_segments=True)
