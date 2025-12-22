"""
data_loader.py - Conflict Matrix Generator
This module loads student-exam data from .stu files, builds a conflict matrix indicating
the number of students enrolled in pairs of exams.
"""

import os
import numpy as np


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_student_data(stu_path):
    students = []
    exam_ids = set()

    with open(stu_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exams = list(map(int, line.split()))
            students.append(exams)
            exam_ids.update(exams)

    return students, sorted(exam_ids)


def build_conflict_matrix(students, exam_ids):
    n_exams = len(exam_ids)

    exam_to_idx = {exam: idx for idx, exam in enumerate(exam_ids)}
    idx_to_exam = {idx: exam for exam, idx in exam_to_idx.items()}

    conflict_matrix = np.zeros((n_exams, n_exams), dtype=int)

    for student_exams in students:
        for i in range(len(student_exams)):
            for j in range(i + 1, len(student_exams)):
                idx_i = exam_to_idx[student_exams[i]]
                idx_j = exam_to_idx[student_exams[j]]
                conflict_matrix[idx_i][idx_j] += 1
                conflict_matrix[idx_j][idx_i] += 1

    return conflict_matrix, exam_to_idx, idx_to_exam


def load_data(dataset_name="car-s-91"):
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data")

    stu_path = os.path.join(data_dir, f"{dataset_name}.stu")

    students, exam_ids = load_student_data(stu_path)

    conflict_matrix, exam_to_idx, idx_to_exam = build_conflict_matrix(students, exam_ids)

    return {
        'students': students,
        'exam_ids': exam_ids,
        'n_exams': len(exam_ids),
        'n_students': len(students),
        'conflict_matrix': conflict_matrix,
        'exam_to_idx': exam_to_idx,
        'idx_to_exam': idx_to_exam
    }


if __name__ == "__main__":
    # Test
    data = load_data("car-s-91")
    print(f"Dataset: car-s-91")
    print(f"Total exams: {data['n_exams']}")
    print(f"Total students: {data['n_students']}")
    print(f"Conflict matrix shape: {data['conflict_matrix'].shape}")
    print(f"Total conflicts: {np.sum(data['conflict_matrix']) // 2}")
