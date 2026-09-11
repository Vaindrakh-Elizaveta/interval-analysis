import numpy as np
import intvalpy as ip

from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


# ============================================================
# НАСТРОЙКИ ВЫВОДА
# ============================================================

np.set_printoptions(
    precision=9,
    suppress=True
)


# ============================================================
# ЧАСТЬ 1. ПРЯМОУГОЛЬНАЯ ИНТЕРВАЛЬНАЯ МАТРИЦА A1
# ============================================================

mid_A1 = np.array([
    [0.95, 1.00],
    [1.05, 1.00],
    [1.10, 1.00]
], dtype=float)


# Первый вариант матрицы радиусов:
# изменяются все элементы
R1_all = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0]
])


# Второй вариант матрицы радиусов:
# изменяется только первый столбец
R1_first_column = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0]
])


def solve_rectangular(mid_A, R):
    """
    Ищет минимальное delta, при котором интервальная
    матрица A = mid_A ± delta * R содержит точечную
    матрицу неполного столбцового ранга.

    Для матрицы 3x2 условие rank(A') < 2 означает,
    что два столбца линейно зависимы:

        A'[:, 0] = lambda * A'[:, 1].
    """

    c1 = mid_A[:, 0]
    c2 = mid_A[:, 1]

    r1 = R[:, 0]
    r2 = R[:, 1]

    # --------------------------------------------------------
    # Для заданного lambda вычисляем минимальное delta,
    # при котором в каждой строке можно выполнить
    #
    # a_i1 = lambda * a_i2
    #
    # При lambda > 0:
    #
    # delta >=
    # |c1_i - lambda*c2_i| /
    # (r1_i + lambda*r2_i)
    #
    # Нужно удовлетворить условию для всех строк,
    # поэтому берём максимум.
    # --------------------------------------------------------

    def delta_for_lambda(lam):

        if lam <= 0:
            return np.inf

        numerator = np.abs(
            c1 - lam * c2
        )

        denominator = (
            r1 + lam * r2
        )

        values = []

        for num, den in zip(
            numerator,
            denominator
        ):

            if np.isclose(den, 0):

                if np.isclose(num, 0):
                    values.append(0.0)

                else:
                    return np.inf

            else:
                values.append(
                    num / den
                )

        return max(values)

    # --------------------------------------------------------
    # Ищем lambda, при котором delta минимально
    # --------------------------------------------------------

    optimization_result = minimize_scalar(
        delta_for_lambda,
        bounds=(1e-6, 10.0),
        method='bounded',
        options={
            'xatol': 1e-13
        }
    )

    lambda_star = optimization_result.x
    delta_star = optimization_result.fun

    # --------------------------------------------------------
    # Создаём интервальную матрицу через intvalpy
    # --------------------------------------------------------

    rad_A = delta_star * R

    A_interval = ip.Interval(
        mid_A,
        rad_A,
        midRadQ=True
    )

    # --------------------------------------------------------
    # Строим точечную матрицу A'
    # --------------------------------------------------------

    first_column = []
    second_column = []

    for i in range(mid_A.shape[0]):

        # Интервал первого элемента строки
        u_left = (
            c1[i]
            - delta_star * r1[i]
        )

        u_right = (
            c1[i]
            + delta_star * r1[i]
        )

        # Интервал lambda * второго элемента строки
        lambda_v_left = (
            lambda_star
            * (
                c2[i]
                - delta_star * r2[i]
            )
        )

        lambda_v_right = (
            lambda_star
            * (
                c2[i]
                + delta_star * r2[i]
            )
        )

        # Пересечение двух интервалов
        left = max(
            u_left,
            lambda_v_left
        )

        right = min(
            u_right,
            lambda_v_right
        )

        # Берём середину пересечения
        common_value = (
            left + right
        ) / 2

        u = common_value
        v = common_value / lambda_star

        first_column.append(u)
        second_column.append(v)

    A_prime = np.column_stack([
        first_column,
        second_column
    ])

    # --------------------------------------------------------
    # Проверки
    # --------------------------------------------------------

    eps = 1e-7

    rank = np.linalg.matrix_rank(
        A_prime,
        tol=eps
    )

    inside = np.all(
        (A_prime >= A_interval.a - eps)
        &
        (A_prime <= A_interval.b + eps)
    )

    dependent = np.allclose(
        A_prime[:, 0],
        lambda_star * A_prime[:, 1],
        atol=eps
    )

    return {
        "delta": delta_star,
        "lambda": lambda_star,
        "A_interval": A_interval,
        "A_prime": A_prime,
        "rank": rank,
        "inside": inside,
        "dependent": dependent
    }


# ============================================================
# ВЫВОД РЕЗУЛЬТАТОВ ДЛЯ A1
# ============================================================

print("=" * 80)
print("ЧАСТЬ 1. ПРЯМОУГОЛЬНАЯ МАТРИЦА A1")
print("=" * 80)


variants_A1 = {
    "Вариант 1: изменяются все элементы":
        R1_all,

    "Вариант 2: изменяется только первый столбец":
        R1_first_column
}


for name, R in variants_A1.items():

    result = solve_rectangular(
        mid_A1,
        R
    )

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    print("\nМатрица центров:")
    print(mid_A1)

    print("\nШаблон матрицы радиусов R:")
    print(R)

    print(
        "\nМинимальное delta =",
        result["delta"]
    )

    print(
        "lambda =",
        result["lambda"]
    )

    print("\nИнтервальная матрица A1:")
    print(
        result["A_interval"]
    )

    print("\nНайденная точечная матрица A1':")
    print(
        result["A_prime"]
    )

    print("\nПроверки:")

    print(
        "rank(A1') =",
        result["rank"]
    )

    print(
        "Столбцы линейно зависимы:",
        result["dependent"]
    )

    print(
        "A1' принадлежит интервальной A1:",
        result["inside"]
    )

    print(
        "Диапазон delta:",
        f"[{result['delta']:.12f}, +inf)"
    )


# ============================================================
# ЧАСТЬ 2. КВАДРАТНАЯ ИНТЕРВАЛЬНАЯ МАТРИЦА A2
# ============================================================

mid_A2 = np.array([
    [1.10, 0.90, 1.10],
    [1.40, 1.00, 0.80],
    [0.80, 1.40, 1.20]
], dtype=float)


R2 = np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
])


def solve_square(
    mid_A,
    R,
    number_of_starts=50,
    random_seed=42
):
    """
    Ищет минимальное delta, при котором интервальная
    квадратная матрица

        A = mid_A ± delta * R

    содержит особенную точечную матрицу A':

        det(A') = 0.

    Представляем:

        A' = mid_A + E,

    где

        |E_ij| <= delta * R_ij.

    Минимизируем delta.
    """

    n = mid_A.shape[0]

    # --------------------------------------------------------
    # Целевая функция:
    # минимизируем delta
    # --------------------------------------------------------

    def objective(z):
        return z[-1]

    # --------------------------------------------------------
    # Условие особенности:
    #
    # det(mid_A + E) = 0
    # --------------------------------------------------------

    def determinant_constraint(z):

        E = z[:-1].reshape(
            n,
            n
        )

        A_prime = (
            mid_A + E
        )

        return np.linalg.det(
            A_prime
        )

    # --------------------------------------------------------
    # Ограничение принадлежности интервальной матрице:
    #
    # |E_ij| <= delta * R_ij
    #
    # scipy требует inequality >= 0,
    # поэтому:
    #
    # delta * R - |E| >= 0
    # --------------------------------------------------------

    def interval_constraints(z):

        E = z[:-1].reshape(
            n,
            n
        )

        delta = z[-1]

        return (
            delta * R
            - np.abs(E)
        ).ravel()

    constraints = [
        {
            'type': 'eq',
            'fun': determinant_constraint
        },
        {
            'type': 'ineq',
            'fun': interval_constraints
        }
    ]

    # --------------------------------------------------------
    # Ограничения на переменные
    #
    # элементы E могут быть любыми,
    # delta >= 0
    # --------------------------------------------------------

    bounds = (
        [(None, None)] * (n * n)
        +
        [(0, None)]
    )

    # --------------------------------------------------------
    # Используем несколько стартов,
    # чтобы уменьшить вероятность попадания
    # в плохой локальный минимум
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    best_result = None

    for k in range(number_of_starts):

        # Случайная стартовая матрица возмущений
        E0 = rng.uniform(
            low=-0.2,
            high=0.2,
            size=(n, n)
        )

        # Стартовое delta.
        # Это только начальное приближение,
        # не ответ задачи.
        delta0 = max(
            np.max(
                np.abs(E0)
            ),
            0.2
        )

        z0 = np.concatenate([
            E0.ravel(),
            [delta0]
        ])

        optimization_result = minimize(
            objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': 1e-13,
                'maxiter': 5000,
                'disp': False
            }
        )

        # ----------------------------------------------------
        # Проверяем корректность найденного решения
        # ----------------------------------------------------

        if optimization_result.success:

            det_error = abs(
                determinant_constraint(
                    optimization_result.x
                )
            )

            interval_error = np.min(
                interval_constraints(
                    optimization_result.x
                )
            )

            if (
                det_error < 1e-7
                and interval_error >= -1e-7
            ):

                if (
                    best_result is None
                    or
                    optimization_result.fun
                    < best_result.fun
                ):

                    best_result = (
                        optimization_result
                    )

    # --------------------------------------------------------
    # Если решение не найдено
    # --------------------------------------------------------

    if best_result is None:

        raise RuntimeError(
            "Не удалось найти допустимую "
            "особенную матрицу."
        )

    # --------------------------------------------------------
    # Получаем итоговые значения
    # --------------------------------------------------------

    delta_star = (
        best_result.x[-1]
    )

    E_star = (
        best_result.x[:-1]
        .reshape(n, n)
    )

    A_prime = (
        mid_A + E_star
    )

    # --------------------------------------------------------
    # Создаём интервальную матрицу через intvalpy
    # --------------------------------------------------------

    rad_A = (
        delta_star * R
    )

    A_interval = ip.Interval(
        mid_A,
        rad_A,
        midRadQ=True
    )

    # --------------------------------------------------------
    # Проверки
    # --------------------------------------------------------

    eps = 1e-7

    inside = np.all(
        (A_prime >= A_interval.a - eps)
        &
        (A_prime <= A_interval.b + eps)
    )

    determinant = np.linalg.det(
        A_prime
    )

    rank = np.linalg.matrix_rank(
        A_prime,
        tol=eps
    )

    return {
        "delta": delta_star,
        "E": E_star,
        "A_interval": A_interval,
        "A_prime": A_prime,
        "determinant": determinant,
        "rank": rank,
        "inside": inside
    }


# ============================================================
# РЕШАЕМ ЗАДАЧУ ДЛЯ A2
# ============================================================

result_A2 = solve_square(
    mid_A2,
    R2,
    number_of_starts=50,
    random_seed=42
)


# ============================================================
# ВЫВОД РЕЗУЛЬТАТОВ ДЛЯ A2
# ============================================================

print("\n\n")
print("=" * 80)
print("ЧАСТЬ 2. КВАДРАТНАЯ МАТРИЦА A2")
print("=" * 80)

print("\nМатрица центров:")
print(mid_A2)

print("\nШаблон матрицы радиусов R:")
print(R2)

print(
    "\nМинимальное delta =",
    result_A2["delta"]
)

print("\nМатрица возмущений E:")
print(
    result_A2["E"]
)

print("\nИнтервальная матрица A2:")
print(
    result_A2["A_interval"]
)

print("\nНайденная особенная точечная матрица A2':")
print(
    result_A2["A_prime"]
)

print("\nПроверки:")

print(
    "det(A2') =",
    result_A2["determinant"]
)

print(
    "rank(A2') =",
    result_A2["rank"]
)

print(
    "A2' принадлежит интервальной A2:",
    result_A2["inside"]
)

print(
    "Диапазон delta:",
    f"[{result_A2['delta']:.12f}, +inf)"
)


# ============================================================
# ИТОГОВОЕ ОБСУЖДЕНИЕ
# ============================================================

print("\n\n")
print("=" * 80)
print("ИТОГ")
print("=" * 80)

result_A1_all = solve_rectangular(
    mid_A1,
    R1_all
)

result_A1_first = solve_rectangular(
    mid_A1,
    R1_first_column
)

print(
    "\nA1, изменяются все элементы:"
)

print(
    "delta_min =",
    result_A1_all["delta"]
)

print(
    "\nA1, изменяется только первый столбец:"
)

print(
    "delta_min =",
    result_A1_first["delta"]
)

print(
    "\nA2, квадратная матрица:"
)

print(
    "delta_min =",
    result_A2["delta"]
)