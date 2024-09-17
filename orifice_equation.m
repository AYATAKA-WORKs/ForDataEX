N_size = 100;
pin = ones(1,N_size)*101325.0*3;
pout= [0:101325*3/(N_size-1):101325.0*3];

for i = 1:size(pin,2)
    phi(i) = fcn(pin(i),pout(i),1.4);
end
plot(pout./pin,phi);

function phi = fcn(pin,pout,kap)
    pr = pout/pin;
    p_cr = (2/(kap+1))^(kap/(kap-1))*pin;
    % Judge critical pressure
    if pout < p_cr
        tmp_phi = sqrt(kap*(2/(kap+1))^((kap+1)/(kap-1)));
    else
        tmp_phi = pr^(1/kap)*sqrt(2*kap/(kap-1)*(1-pr^((kap-1)/kap)));
    end
phi = tmp_phi;
end