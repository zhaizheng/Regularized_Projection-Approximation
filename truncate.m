
function A = truncate(A, mu)
    A(A<mu) = 0;
end

