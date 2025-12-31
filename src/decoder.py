import numpy as np
from typing import List, Dict, Tuple

def decode_greedy(permutation: List[int], conflict_matrix: np.ndarray, exam_to_idx: Dict[int, int], max_slots: int = None) -> Tuple[Dict[int, int], int]:
    slots, slot_assignment = [], {}
    for exam_id in permutation:
        exam_idx = exam_to_idx[exam_id]
        assigned = False
        for slot_num, slot_exams in enumerate(slots):
            has_conflict = False
            for other_idx in slot_exams:
                if conflict_matrix[exam_idx][other_idx] > 0:
                    has_conflict = True; break
            if not has_conflict:
                if max_slots is None or slot_num < max_slots:
                    slots[slot_num].add(exam_idx); slot_assignment[exam_id] = slot_num; assigned = True; break
        if not assigned:
            if max_slots is None or len(slots) < max_slots:
                slots.append({exam_idx}); slot_assignment[exam_id] = len(slots) - 1
            else:
                slot_assignment[exam_id] = force_assign(exam_idx, slots, conflict_matrix)
    return slot_assignment, len(slots)

def force_assign(exam_idx: int, slots: List[set], conflict_matrix: np.ndarray) -> int:
    min_conflicts, best_slot = float('inf'), 0
    for slot_num, slot_exams in enumerate(slots):
        conflicts = sum(conflict_matrix[exam_idx][other_idx] for other_idx in slot_exams)
        if conflicts < min_conflicts: min_conflicts, best_slot = conflicts, slot_num
    slots[best_slot].add(exam_idx)
    return best_slot

def get_slot_exams(slot_assignment: Dict[int, int]) -> Dict[int, List[int]]:
    slot_exams = {}
    for exam_id, slot_num in slot_assignment.items():
        if slot_num not in slot_exams: slot_exams[slot_num] = []
        slot_exams[slot_num].append(exam_id)
    return slot_exams

def count_hard_violations(slot_assignment: Dict[int, int], conflict_matrix: np.ndarray, exam_to_idx: Dict[int, int]) -> int:
    slot_exams = get_slot_exams(slot_assignment)
    violations = 0
    for slot_num, exams in slot_exams.items():
        for i in range(len(exams)):
            for j in range(i + 1, len(exams)):
                idx_i, idx_j = exam_to_idx[exams[i]], exam_to_idx[exams[j]]
                if conflict_matrix[idx_i][idx_j] > 0: violations += conflict_matrix[idx_i][idx_j]
    return violations

if __name__ == "__main__":
    import random
    from data_loader import load_data
    data = load_data("car-s-91")
    permutation = data['exam_ids'].copy()
    random.shuffle(permutation)
    slot_assignment, n_slots = decode_greedy(permutation, data['conflict_matrix'], data['exam_to_idx'])
    violations = count_hard_violations(slot_assignment, data['conflict_matrix'], data['exam_to_idx'])
    print(f"Decoder Test (car-s-91)\nTotal exams: {len(permutation)}\nUsed slots: {n_slots}\nHard constraint violations: {violations}")
