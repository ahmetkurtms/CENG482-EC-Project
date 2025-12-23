"""
fitness.py - Fitness Function for Exam Timetabling
----------------------------------------------------
This module implements the fitness function for evaluating exam timetabling solutions.

NOTE: hard_violations is always 0 when max_slots=None (current setting).
      hard_penalty is only relevant if max_slots limit is enforced.
"""

import numpy as np
from typing import Dict, List, Tuple
from decoder import decode_greedy, count_hard_violations, get_slot_exams


def calculate_proximity_cost(slot_assignment: Dict[int, int],
                            students: List[List[int]],
                            exam_to_idx: Dict[int, int],
                            weights: List[int] = None) -> float:
    if weights is None:
        weights = [16, 8, 4, 2, 1]  # Carter standard weights

    total_penalty = 0

    for student_exams in students:
        exam_slots = []
        for exam_id in student_exams:
            if exam_id in slot_assignment:
                exam_slots.append((exam_id, slot_assignment[exam_id]))

        for i in range(len(exam_slots)):
            for j in range(i + 1, len(exam_slots)):
                slot_diff = abs(exam_slots[i][1] - exam_slots[j][1])

                if 1 <= slot_diff <= len(weights):
                    total_penalty += weights[slot_diff - 1]

    n_students = len(students)
    return total_penalty / n_students if n_students > 0 else 0


def fitness_function(permutation: List[int],
                    conflict_matrix: np.ndarray,
                    exam_to_idx: Dict[int, int],
                    students: List[List[int]],
                    max_slots: int = None) -> Tuple[float, Dict]:
                    # hard_penalty: float = 1000000.0  # NOTE: Uncomment if max_slots limit is used
    slot_assignment, n_slots = decode_greedy(
        permutation, conflict_matrix, exam_to_idx, max_slots
    )

    # NOTE: hard_violations is always 0 when max_slots=None (current setting)
    # hard_violations = count_hard_violations(
    #     slot_assignment, conflict_matrix, exam_to_idx
    # )
    hard_violations = 0  # Always 0 with unlimited slots

    proximity_cost = calculate_proximity_cost(
        slot_assignment, students, exam_to_idx
    )

    # NOTE: With max_slots=None, fitness = proximity_cost (no hard penalty needed)
    # fitness = proximity_cost + (hard_violations * hard_penalty)
    fitness = proximity_cost

    details = {
        'slot_assignment': slot_assignment,
        'n_slots': n_slots,
        'hard_violations': hard_violations,
        'proximity_cost': proximity_cost,
        'fitness': fitness
    }

    return fitness, details


def evaluate_population(population: List[List[int]],
                        conflict_matrix: np.ndarray,
                        exam_to_idx: Dict[int, int],
                        students: List[List[int]],
                        max_slots: int = None) -> List[Tuple[float, Dict]]:
    results = []
    for individual in population:
        fitness, details = fitness_function(
            individual, conflict_matrix, exam_to_idx, students, max_slots
        )
        results.append((fitness, details))
    return results


if __name__ == "__main__":
    # Test
    import random
    from data_loader import load_data

    data = load_data("car-s-91")

    permutation = data['exam_ids'].copy()
    random.shuffle(permutation)

    fitness, details = fitness_function(
        permutation,
        data['conflict_matrix'],
        data['exam_to_idx'],
        data['students']
    )

    print(f"Fitness Test (car-s-91)")
    print(f"Slots used: {details['n_slots']}")
    print(f"Hard constraint violations: {details['hard_violations']}")
    print(f"Proximity cost: {details['proximity_cost']:.4f}")
    print(f"Total fitness: {details['fitness']:.4f}")
