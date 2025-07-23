import heapq
import math
import torch

class APQMem:
    """
    Adaptive Priority-Queue Memory (APQ-Mem)
    An advanced memory bank for Test-Time Adaptation that improves upon CSTU with:
    1.  Efficiency: Uses a Priority Queue (heap) for O(log K) removal operations.
    2.  Adaptability: Implements an adaptive aging mechanism that responds to domain drifts.
    3.  Awareness: A new heuristic score that penalizes over-represented classes.
    """
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=0.5, age_factor_bonus=5.0):
        # Core parameters
        self.capacity = capacity
        self.num_class = num_class
        self.per_class_capacity = self.capacity / self.num_class

        # Hyperparameters for heuristic score and aging
        self.lambda_t = lambda_t  # Weight for timeliness
        self.lambda_u = lambda_u  # Weight for uncertainty
        self.lambda_d = lambda_d  # Weight for distribution penalty (NEW)
        self.age_factor_bonus = age_factor_bonus # Bonus for adaptive aging speed (NEW)

        # Data structure: A list of priority queues (min-heaps)
        # We store (-score, item_id, item_data) to simulate a max-heap for scores
        self.data: list[list] = [[] for _ in range(self.num_class)]
        self.item_id_counter = 0 # To break ties in the priority queue

    def get_occupancy(self) -> int:
        """Returns the total number of items in the memory bank."""
        return sum(len(class_heap) for class_heap in self.data)

    def per_class_dist(self) -> list[int]:
        """Returns a list with the number of items per class."""
        return [len(class_heap) for class_heap in self.data]

    def add_instance(self, instance, drift_signal=0.0):
        """
        Main method to add a new instance to the memory bank.
        It handles the logic of removing an old instance if the bank is full.
        """
        x, prediction, uncertainty = instance
        
        # Calculate the score for the new item
        # Since it's new, age is 0 and it doesn't get a distribution penalty yet
        new_score = self._heuristic_score(0, uncertainty, prediction, is_new=True)

        # Determine if we need to remove an old instance
        should_add = self._should_add(prediction, new_score)
        
        if should_add:
            # Add the new item to the corresponding class heap
            new_item_tuple = (-new_score, self.item_id_counter, (x, uncertainty, 0))
            heapq.heappush(self.data[prediction], new_item_tuple)
            self.item_id_counter += 1

        # Apply adaptive aging to all existing items in the memory bank
        self._adaptive_age(drift_signal)

    def _should_add(self, cls: int, new_score: float) -> bool:
        """
        Determines if a new item should be added, potentially by removing an existing one.
        This function contains the core replacement logic.
        """
        class_occupancy = len(self.data[cls])
        total_occupancy = self.get_occupancy()

        # Case 1: The memory bank is not full yet. Always add.
        if total_occupancy < self.capacity:
            return True

        # Case 2: The memory bank is full. We must remove one item to add a new one.
        # Determine the search space for the item to be removed.
        if class_occupancy < self.per_class_capacity:
            # If the new item's class is under-represented, remove from an over-represented class.
            search_classes = self._get_majority_classes()
        else:
            # If the new item's class is already full, try to remove from this class.
            search_classes = [cls]
        
        # Find the worst item (highest score) in the search space
        worst_item_score, worst_item_class = self._find_worst_item(search_classes)

        if worst_item_score is not None and worst_item_score > new_score:
            # If the worst item is "worse" than the new item, remove it.
            # heapq.heappop removes the item with the smallest value, which is the one with the highest score
            # because we stored -score.
            heapq.heappop(self.data[worst_item_class])
            return True # A spot has been freed up.
        else:
            # The new item is not good enough to replace any existing item.
            return False

    def _find_worst_item(self, classes_to_search: list[int]):
        """Finds the item with the highest heuristic score among a list of classes."""
        worst_score = -float('inf')
        worst_class = None

        for cls in classes_to_search:
            if not self.data[cls]:
                continue
            
            # The item with the highest score is at the root of the heap
            # because we stored (-score). So, its score is -heap[0][0].
            current_worst_score_in_class = -self.data[cls][0][0]
            if current_worst_score_in_class > worst_score:
                worst_score = current_worst_score_in_class
                worst_class = cls
        
        return worst_score, worst_class

    def _get_majority_classes(self) -> list[int]:
        """Finds the class(es) with the most items."""
        per_class_distribution = self.per_class_dist()
        if not per_class_distribution:
            return []
        max_occupied = max(per_class_distribution)
        return [i for i, occupied in enumerate(per_class_distribution) if occupied == max_occupied]

    def _heuristic_score(self, age: float, uncertainty: float, cls: int, is_new: bool = False) -> float:
        """
        Calculates the heuristic score for an item. A higher score means the item
        is a better candidate for removal.
        """
        # Part 1: Timeliness score (sigmoid of age)
        # Normalized age before sigmoid to prevent overflow and provide better scaling
        norm_age = age / self.capacity 
        timeliness_score = 1 / (1 + math.exp(-norm_age))

        # Part 2: Uncertainty score (normalized entropy)
        uncertainty_score = uncertainty / math.log(self.num_class) if self.num_class > 1 else uncertainty

        # Part 3: Distribution penalty (NEW)
        # This penalty is only applied to existing items, not the new candidate item.
        distribution_penalty = 0.0
        if not is_new:
            class_occupancy = len(self.data[cls])
            # Penalize only if the class occupancy exceeds its fair share
            if class_occupancy > self.per_class_capacity:
                distribution_penalty = (class_occupancy - self.per_class_capacity) / self.capacity

        return (self.lambda_t * timeliness_score +
                self.lambda_u * uncertainty_score +
                self.lambda_d * distribution_penalty)

    def _adaptive_age(self, drift_signal: float):
        """Increases the age of all items, with a speed adapted to the domain drift."""
        # Calculate the aging speed. It's 1 in stable conditions, and higher during drifts.
        # relu ensures that we only speed up aging, not slow it down.
        aging_speed = 1.0 + self.age_factor_bonus * max(0, drift_signal)

        # Update all items in the memory bank
        for cls in range(self.num_class):
            updated_heap = []
            for neg_score, item_id, (data, uncertainty, age) in self.data[cls]:
                new_age = age + aging_speed
                # Recalculate score with the new age to maintain heap property
                new_score = self._heuristic_score(new_age, uncertainty, cls)
                updated_heap.append((-new_score, item_id, (data, uncertainty, new_age)))
            
            # Re-build the heap for the class
            heapq.heapify(updated_heap)
            self.data[cls] = updated_heap

    def get_memory(self) -> tuple[list, list]:
        """Returns all data and their normalized ages from the memory bank."""
        all_data = []
        all_ages = []

        for class_heap in self.data:
            for _, _, (data, _, age) in class_heap:
                all_data.append(data)
                # Normalize age for the re-weighting loss function
                all_ages.append(age / self.capacity)

        return all_data, all_ages