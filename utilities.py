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


class Vector(object):
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def __matmul__(self, other: "Vector") -> int:
        return self._x * other._x + self._y * other._y

    def __rmatmul__(self, other: "Vector") -> int:
        return other._x * self._x + other._y * self._y


class Triangle(object):
    def __init__(
        self,
        vertex_a: tuple[int, int],
        vertex_b: tuple[int, int],
        vertex_c: tuple[int, int],
    ) -> None:
        self._x_a, self._y_a = vertex_a
        x_b, y_b = vertex_b
        x_c, y_c = vertex_c
        self._v_ac = Vector(x_c - self._x_a, y_c - self._y_a)
        self._v_ab = Vector(x_b - self._x_a, y_b - self._y_a)
        self._p_ab_ab = self._v_ab @ self._v_ab
        self._p_ac_ac = self._v_ac @ self._v_ac
        self._p_ab_ac = self._v_ab @ self._v_ac
        self._p_ac_ab = self._v_ac @ self._v_ab

    def __contains__(self, point: tuple[int, int]) -> bool:
        x_p, y_p = point
        v_ap = Vector(x_p - self._x_a, y_p - self._y_a)
        p_ap_ab, p_ap_ac = v_ap @ self._v_ab, v_ap @ self._v_ac
        if (
            p_ap_ac * self._p_ab_ab - p_ap_ab * self._p_ac_ab >= 0
            and p_ap_ab * self._p_ac_ac - p_ap_ac * self._p_ab_ac >= 0
            and (
                (p_ap_ac * self._p_ab_ab - p_ap_ab * self._p_ac_ab)
                + (p_ap_ab * self._p_ac_ac - p_ap_ac * self._p_ab_ac)
                - (self._p_ac_ac * self._p_ab_ab - self._p_ac_ab * self._p_ac_ab)
            )
            <= 0
        ):
            return True
        return False
