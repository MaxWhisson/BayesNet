# module of helper functions for other modules
module HelperFunctions

export  propagate_matrix_opp,
        propagate_vector_opp,
        set_col_matrix_expr,
        set_sub_array_expr,
        triangular,
        to_lower_triangular,
        trace,
        set_sub_vector_expr,
        init_L_diagonal_cov

using Zygote

# applies f to column i of matrix2 and assigns to column i + 1 of matrix1
function propagate_matrix_opp(matrix1, matrix2, i, f)                                                                                                                                                                                                                                                 
    matrix1[:,i + 1] = f(matrix2[:, i])                                                                                                                                                                                                                     
    return matrix1
end

# applies f to column i of matrix2 and assigns to element i + 1 of vector1
function propagate_vector_opp(vector1, matrix2, i, f)                                                                                                                                                                                                                                                 
    vector1[i + 1] = f(matrix2[:,i])                                                                                                                                                                                                                     
    return vector1
end

# set column of matrix and return matrix
function set_col_matrix_expr(matrix, i, col)
    matrix[:,i] = col
    return matrix
end

# set portion of matrix to sub matrix
function set_sub_array_expr(i1, i2, j1, j2, newSubArr, matrix)
    matrix[i1:i2, j1:j2] = newSubArr
    return matrix
end

# get nth triangular number
function triangular(n)
    sum(1:n)
end

# safe log for numerical stability
function s_log(x)
    log(x == 0 ? x + eps() : x)
end

# turn vector into lower triangular matrix
function to_lower_triangular(arr, D)
    (L, next_i) = foldl(
        ((M, next_j), i) -> (
            set_sub_array_expr(i, i, 1, i, arr[next_j:next_j + i - 1], M), 
            next_j + i
        ),
        1:D, 
        init = (zeros(D, D), 1)
    )
    return L
end

# turn lower triangular matrix into vector
function flatten_triangular(L, D)
    foldl(
        ((flattened, next_i), i) -> (
            set_sub_vector_expr(next_i, next_i + i - 1, flattened, L[i,1:i]),
            next_i + i
        ),
        1:D,
        init = (zeros(triangular(D)), 1)
    )[1]
end

# custom adjoint for to_lower_triangular()
@Zygote.adjoint to_lower_triangular(arr, D) = (
    to_lower_triangular(arr, D),
    L′ -> (flatten_triangular(L′, D), 0)
)

# initialise lower triangular diagonal matrix
function init_L_diagonal_cov(D)
    init_L = -20ones(triangular(D))
    t = 0
    for i in 1:D
        t += i
        init_L[t] = -10log(ℯ - 1)
    end
    init_L
end

# set part of array to new sub array and return
function set_sub_vector_expr(i1, i2, arr, new_sub_arr)
    arr[i1:i2] = new_sub_arr
    arr
end

# functional style log
function trace(x, y)
    println(x)
    return y
end

# functional style log
function trace(x)
    println(x)
    return x
end

end