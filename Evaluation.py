import pandas as pd
import numpy as np


def scale_coefficients(weights, mean_weight):
    scaled_weights = [0.5 * (weight + mean_weight) for weight in weights]
    return scaled_weights


def linear_transform(weights, target_min=0.6, target_max=1.0):
    transformed_weights = [
        target_min + (weight - 0) * (target_max - target_min) / (0.9664031620553359 - 0)
        for weight in weights
    ]
    return transformed_weights


def crop_data(data, regex):
    return data.filter(regex=regex)


def filter_data(data, indices_to_remove):
    data = data.drop(indices_to_remove)
    return data


def filter_parameters(SST, SSE, SSR, indices_to_remove):
    SST = [sst for i, sst in enumerate(SST) if i not in indices_to_remove]
    SSE = [sse for i, sse in enumerate(SSE) if i not in indices_to_remove]
    SSR = [ssr for i, ssr in enumerate(SSR) if i not in indices_to_remove]
    return SST, SSE, SSR


if __name__ == "__main__":
    calibration_data = pd.read_csv("calib_grades.csv")
    student_grades = crop_data(calibration_data, r"q(1[0-7]|[1-9])_calib$")
    correct_grades = crop_data(calibration_data, r"q(1[0-7]|[1-9])_calib_crct$")
    correct_grades.columns = student_grades.columns

    # Faza 1
    SST = [sum((row - row.mean()) ** 2) for _, row in student_grades.iterrows()]
    SSE = [sum((row - correct) ** 2)
           for (_, row), (_, correct) in zip(student_grades.iterrows(), correct_grades.iterrows())]
    SSR = [sst - sse if sst != 0 else None for sst, sse in zip(SST, SSE)]

    zero_idx = [i for i, sst in enumerate(SST) if sst == 0]
    calibration_data = filter_data(calibration_data, zero_idx)
    SST, SSE, SSR = filter_parameters(SST, SSE, SSR, zero_idx)

    nan_idx = [i for i, ssr in enumerate(SSR) if pd.isna(ssr)]
    calibration_data = filter_data(calibration_data, nan_idx)
    SST, SSE, SSR = filter_parameters(SST, SSE, SSR, nan_idx)

    weights = [1 - (sse / sst) for sse, sst in zip(SSE, SST)]  # rsquared

    mean_weight = sum(weights) / len(weights)
    scaled_weights = scale_coefficients(weights, mean_weight)

    transformed_weights = linear_transform(weights, target_min=0.6, target_max=1.0)
    print(weights)
    print(transformed_weights)

    calibration_data["weights"] = scaled_weights
    calibration_data.to_csv("calib_grades.csv", index=False)

    peer_review_data = pd.read_csv("peer_review_grades.csv")
    first_student = crop_data(peer_review_data, r"q(1[0-7]|[1-9])_1$")
    second_student = crop_data(peer_review_data, r"q(1[0-7]|[1-9])_2$")
    third_student = crop_data(peer_review_data, r"q(1[0-7]|[1-9])_3$")
    fourth_student = crop_data(peer_review_data, r"q(1[0-7]|[1-9])_4$")

    first_student_grades = [row.mean(skipna=True) for _, row in first_student.iterrows()]
    second_student_grades = [row.mean(skipna=True) for _, row in second_student.iterrows()]
    third_student_grades = [row.mean(skipna=True) for _, row in third_student.iterrows()]
    fourth_student_grades = [row.mean(skipna=True) for _, row in fourth_student.iterrows()]

    peer_review_data["first_student"] = first_student_grades
    peer_review_data["second_student"] = second_student_grades
    peer_review_data["third_student"] = third_student_grades
    peer_review_data["fourth_student"] = fourth_student_grades
    peer_review_data.to_csv("peer_review_grades.csv", index=False)

    jobs_students = pd.read_excel("PA 2019_20.xlsx", sheet_name="jobs-student")
    dict_students = {}

    for _, row in peer_review_data.iterrows():
        jobs = [job for job in row["jobs"].replace("{", "").replace("}", "").split(",") if int(job) > 0]
        grader = row["id_student"]
        for idx, job in enumerate(jobs):
            student = jobs_students[jobs_students["id_job"] == int(job)]["id_student"].values[0]

            if student not in dict_students:
                dict_students[student] = []

            grade = row.iloc[idx - 4]
            try:
                weight = calibration_data[calibration_data["id_student"] == grader]["weights"].values[0]
            except IndexError:
                break

            dict_students[student].append((grade, weight))

    students_calculated_grades = {}
    for student, grades in dict_students.items():
        students_calculated_grades[student] = sum([grade * weight for grade, weight in grades]) / sum(
            [weight for _, weight in grades])

    real_grades = pd.read_excel("PA 2019_20.xlsx", sheet_name="Faza1 – ZPR ocjene ")

    comparison_data = pd.DataFrame()
    comparison_data["id_student"] = students_calculated_grades.keys()
    comparison_data["calculated_grade"] = students_calculated_grades.values()
    comparison_data["calculated_grade_rounded"] = comparison_data["calculated_grade"].round()
    comparison_data = comparison_data.merge(real_grades, on="id_student")
    accuracy = ((comparison_data["calculated_grade_rounded"] == comparison_data["bodovi"].round()).sum() / len(
        comparison_data))
    print(accuracy)

    comparison_data.rename(columns={"bodovi": "real_grade"}, inplace=True)
    comparison_data.to_csv("comparison.csv", index=False)

    # Faza 2
    student_grading_grades = pd.read_excel("PA 2019_20.xlsx", sheet_name="Faza2 – ZPR ocjene ")
    new_rows = []
    for idx, row in calibration_data.iterrows():
        student = row["id_student"]
        coefficient = abs(transformed_weights[idx])
        pts_received = student_grading_grades[student_grading_grades["id_student"] == student]["bodovi"].values[0]
        prediction = coefficient * 5
        new_row = {
            "id_student": student,
            "bodovi": pts_received,
            "koef": coefficient,
            # "prediction": prediction,
            "grade": float(round(prediction))  # grade_decision(prediction)
        }
        new_rows.append(new_row)

    student_grading_grades_compared = pd.DataFrame(new_rows)
    accuracy = ((student_grading_grades_compared["grade"] == student_grading_grades_compared["bodovi"]).sum() / len(
        student_grading_grades_compared))
    print(accuracy)
    student_grading_grades_compared.to_csv("student_grading_grades_compared.csv", index=False)
