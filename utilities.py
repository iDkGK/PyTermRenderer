from hintings import MatrixType, MatrixRowType


def matrix_transpose(matrix: MatrixType) -> MatrixType:
    return list(map(list, zip(*matrix)))


def matrix_add(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    result_matrix: MatrixType = []
    # deep copy to avoid modification of first matrix
    for first_matrix_row in first_matrix:
        result_matrix_row: MatrixRowType = []
        for element in first_matrix_row:
            result_matrix_row.append(element)
        result_matrix.append(result_matrix_row)
    # continuously addition
    for other_matrix in matrices_iterator:
        for row_index, (result_matrix_row, other_matrix_row) in enumerate(
            zip(result_matrix, other_matrix)
        ):
            for element_index in range(0, len(result_matrix_row)):
                result_matrix[row_index][element_index] = (  # type: ignore
                    result_matrix_row[element_index] + other_matrix_row[element_index]
                )
    return result_matrix


def matrix_subtract(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    result_matrix: MatrixType = []
    # deep copy to avoid modification of first matrix
    for first_matrix_row in first_matrix:
        result_matrix_row: MatrixRowType = []
        for element in first_matrix_row:
            result_matrix_row.append(element)
        result_matrix.append(result_matrix_row)
    # continuously subtraction
    for other_matrix in matrices_iterator:
        for row_index, (result_matrix_row, other_matrix_row) in enumerate(
            zip(result_matrix, other_matrix)
        ):
            for element_index in range(0, len(result_matrix_row)):
                result_matrix[row_index][element_index] = (  # type: ignore
                    result_matrix_row[element_index] - other_matrix_row[element_index]
                )
    return result_matrix


def matrix_multiply(*matrices: MatrixType) -> MatrixType:
    matrices_iterator = iter(matrices)
    first_matrix = next(matrices_iterator)
    result_matrix: MatrixType = []
    # deep copy to avoid modification of first matrix
    for first_matrix_row in first_matrix:
        result_matrix_row: MatrixRowType = []
        for element in first_matrix_row:
            result_matrix_row.append(element)
        result_matrix.append(result_matrix_row)
    # continuously multiplication
    for other_matrix in matrices_iterator:
        other_matrix = matrix_transpose(other_matrix)
        result_matrix = [
            [
                sum(
                    result_matrix_element * other_matrix_element
                    for result_matrix_element, other_matrix_element in zip(row, column)
                )
                for column in other_matrix
            ]
            for row in result_matrix
        ]
    return result_matrix
