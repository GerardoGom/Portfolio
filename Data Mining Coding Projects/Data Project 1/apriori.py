
import sys
import time
import numpy as np
from collections import defaultdict
from itertools import combinations


class AprioriMiner:

    
    def __init__(self, min_support):
        self.min_support = min_support
        self.transactions = []
        self.frequent_itemsets = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def load_transactions(self, filename):
       #Load transactions from file.
        print(f"Loading transactions from {filename}...")
        start = time.time()
        
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split()
                if items:
                    self.transactions.append([int(x) for x in items])
        
        print(f"Loaded {len(self.transactions)} transactions in {time.time()-start:.2f}s")
    
    def create_bitmap_matrix(self, transactions, frequent_items):
        
        #Create a bitmap matrix for fast support counting.
        #Rows = transactions, Columns = items
        
        # Map items to indices
        self.idx_to_item = sorted(list(frequent_items))
        self.item_to_idx = {item: idx for idx, item in enumerate(self.idx_to_item)}
        
        n_trans = len(transactions)
        n_items = len(frequent_items)
        
        # Create binary matrix
        matrix = np.zeros((n_trans, n_items), dtype=np.bool_)
        
        for i, trans in enumerate(transactions):
            for item in trans:
                if item in self.item_to_idx:
                    matrix[i, self.item_to_idx[item]] = True
        
        return matrix
    
    def get_frequent_1_itemsets(self):
       # Find frequent 1-itemsets.
        print("Finding frequent 1-itemsets...")
        start = time.time()
        
        # Count each item
        item_counts = defaultdict(int)
        for trans in self.transactions:
            for item in trans:
                item_counts[item] += 1
        
        # Filter frequent items
        frequent_items = set()
        frequent_1 = {}
        
        for item, count in item_counts.items():
            if count >= self.min_support:
                frequent_items.add(item)
                frequent_1[frozenset([item])] = count
        
        print(f"Found {len(frequent_1)} frequent 1-itemsets in {time.time()-start:.2f}s")
        
        # Prune transactions
        self.transactions = [
            [item for item in trans if item in frequent_items]
            for trans in self.transactions
        ]
        self.transactions = [t for t in self.transactions if t]
        
        print(f"Transactions after pruning: {len(self.transactions)}")
        
        return frequent_1, frequent_items
    
    def generate_candidates(self, prev_itemsets, k):
        
      #  Generate k-item candidates from (k-1)-item frequent itemsets.
      #  Uses F_{k-1} x F_{k-1} method with lexicographic ordering.
        
        candidates = []
        n = len(prev_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                l1 = prev_itemsets[i]
                l2 = prev_itemsets[j]
                
                # Merge if first k-2 items match
                if k == 2:
                    candidates.append((l1[0], l2[0]))
                elif l1[:k-2] == l2[:k-2]:
                    # Create candidate
                    candidate = l1[:k-1] + (l2[k-2],)
                    candidates.append(candidate)
        
        return candidates
    
    def prune_candidates(self, candidates, prev_set, k):
      #  Prune candidates using Apriori property.
        if k == 2:
            return candidates
        
        pruned = []
        for cand in candidates:
            # Check if all (k-1)-subsets are frequent
            is_valid = True
            for i in range(k):
                subset = cand[:i] + cand[i+1:]
                if subset not in prev_set:
                    is_valid = False
                    break
            if is_valid:
                pruned.append(cand)
        
        return pruned
    
    def count_support_fast(self, candidates, matrix, k):
        
      #  Fast support counting using bitmap matrix with batched operations.
        
        result = {}
        
        # Process candidates in batches for better cache performance
        batch_size = 1000
        n_candidates = len(candidates)
        
        for batch_start in range(0, n_candidates, batch_size):
            batch_end = min(batch_start + batch_size, n_candidates)
            batch = candidates[batch_start:batch_end]
            
            for cand in batch:
                # Get column indices for items in candidate
                indices = [self.item_to_idx[item] for item in cand]
                
                # Compute AND across these columns
                # Start with first column, then AND with others
                mask = matrix[:, indices[0]]
                for idx in indices[1:]:
                    mask = mask & matrix[:, idx]
                
                support = np.sum(mask)
                
                if support >= self.min_support:
                    result[frozenset(cand)] = int(support)
        
        return result
    
    def mine(self):
      #  Execute Apriori algorithm.
        print(f"\nApriori Mining (min_support = {self.min_support})")
        print("=" * 60)
        overall_start = time.time()
        
        # Get frequent 1-itemsets
        frequent_k, frequent_items = self.get_frequent_1_itemsets()
        self.frequent_itemsets.update(frequent_k)
        
        # Create bitmap matrix for fast counting
        print("Creating bitmap matrix...")
        start = time.time()
        matrix = self.create_bitmap_matrix(self.transactions, frequent_items)
        print(f"Matrix created: {matrix.shape} in {time.time()-start:.2f}s")
        
        # Prepare for iteration
        frequent_list = [tuple(sorted(fs)) for fs in frequent_k.keys()]
        frequent_list.sort()
        
        k = 2
        
        # Main loop
        while frequent_list:
            print(f"\nFinding frequent {k}-itemsets...")
            iter_start = time.time()
            
            # Prune transactions too small for k-itemsets
            if k > 2:
                row_sums = np.sum(matrix, axis=1)
                valid_rows = row_sums >= k
                if np.sum(valid_rows) < len(matrix):
                    matrix = matrix[valid_rows]
                    print(f"  Pruned to {len(matrix)} transactions")
                
                if len(matrix) == 0:
                    print("  No transactions left. Stopping.")
                    break
            
            # Generate candidates
            candidates = self.generate_candidates(frequent_list, k)
            if not candidates:
                print("  No candidates generated. Stopping.")
                break
            
            # Prune using Apriori property
            prev_set = set(frequent_list)
            candidates = self.prune_candidates(candidates, prev_set, k)
            print(f"  Candidates: {len(candidates)}")
            
            if not candidates:
                print("  All pruned. Stopping.")
                break
            
            # Count support
            frequent_k = self.count_support_fast(candidates, matrix, k)
            
            elapsed = time.time() - iter_start
            print(f"  Frequent {k}-itemsets: {len(frequent_k)} ({elapsed:.2f}s)")
            
            if not frequent_k:
                break
            
            self.frequent_itemsets.update(frequent_k)
            
            # Prepare next iteration
            frequent_list = [tuple(sorted(fs)) for fs in frequent_k.keys()]
            frequent_list.sort()
            
            k += 1
        
        total_time = time.time() - overall_start
        print("\n" + "=" * 60)
        print(f"Mining completed in {total_time:.2f} seconds")
        print(f"Total frequent itemsets: {len(self.frequent_itemsets)}")
        print("=" * 60)
    
    def save_results(self, output_file):
       # Save results to file.
        print(f"\nSaving to {output_file}...")
        
        sorted_itemsets = sorted(
            self.frequent_itemsets.items(),
            key=lambda x: (len(x[0]), sorted(x[0]))
        )
        
        with open(output_file, 'w') as f:
            for itemset, count in sorted_itemsets:
                items = sorted(list(itemset))
                f.write(f"{' '.join(map(str, items))} ({count})\n")
        
        print(f"Saved {len(self.frequent_itemsets)} itemsets")


def main():
    if len(sys.argv) != 4:
        print("Usage: python apriori.py <input_file> <min_support> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        min_support = int(sys.argv[2])
        if min_support <= 0:
            raise ValueError()
    except:
        print("Error: min_support must be a positive integer")
        sys.exit(1)
    
    output_file = sys.argv[3]
    
    print("=" * 60)
    print("APRIORI ALGORITHM")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"MinSup: {min_support}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    try:
        miner = AprioriMiner(min_support)
        miner.load_transactions(input_file)
        miner.mine()
        miner.save_results(output_file)
        print("\nExecution completed successfully!")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
