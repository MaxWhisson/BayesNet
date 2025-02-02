module HelperFunctions

export  propagate_matrix_opp,
        set_col_matrix_expr,
        set_sub_array_expr,
        triangular,
        to_lower_triangular,
        trace

function propagate_matrix_opp(matrix1, matrix2, i, f)                                                                                                                                                                                                                                                 
    matrix1[:,i + 1] = f(matrix2[:, i])                                                                                                                                                                                                                     
    return matrix1
end

function set_col_matrix_expr(matrix, i, col)
    matrix[:,i] = col
    return matrix
end

function set_sub_array_expr(i1, i2, j1, j2, newSubArr, matrix)
    matrix[i1:i2, j1:j2] = newSubArr
    return matrix
end

function triangular(n)
    sum(1:n)
end

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

function trace(x, y)
    println(x)
    return y
end

end