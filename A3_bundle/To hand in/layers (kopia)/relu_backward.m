function dldx = relu_backward(x, dldy)
    % "The value heaviside(0) is 0.5 by default. It
    %  can be changed to any value v by the call 
    %  sympref('HeavisideAtOrigin', v)."
    sympref('HeavisideAtOrigin', 0);
    dldx = dldy.*heaviside(x);
end
