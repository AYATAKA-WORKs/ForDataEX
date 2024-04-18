clear all
% Turbin operational condition
Pi = 250.000;   % kPa
Po = 101.325;   % kPa
Ti = 100.0;     % degC
RHi = 20;       % %
eff_tbn = 0.9;

% Calc moist air property
Pwsi = calc_sat_pressure(Ti);   % kPa
Pwi = RHi/100*Pwsi;             % kPa
Wi = 0.62198*Pwi/(Pi-Pwi);     % kg/kg

% calc adiabatic expansion
To_dry = Ti - eff_tbn*(Ti+273.15)*(1-(Pi/Po)^(-(1.4-1)/1.4));
ho_dry = calc_entalpy_air(To_dry, 1.005, 1.805, Wi, 2501.0);

% 凝縮ジャッジ
Pwo = Pwi/Pi*Po;
Pwso = calc_sat_pressure(To_dry);

if Pwo > Pwso
    disp("凝縮！");
    % Solve turbine outlet temperature by Newton–Raphson method
    % Solver setting
    iteration = 0;
    delta = 1e-10;
    error_min = 1e-5;
    error = 1;
    max_iteration = 999;
    % Initial state
    To = To_dry;
    
    while error > error_min && iteration < max_iteration
        Pwso = calc_sat_pressure(To);
        Wo = 0.62198*Pwso/(Po-Pwso);
        ho = calc_entalpy_mix(To, 1.005, 1.805, 4.186, Wi, Wo, 2501.0);
        fn = ho_dry - ho;
        Pwso_delta = calc_sat_pressure(To+delta);
        Wo_delta = 0.62198*Pwso_delta/(Po-Pwso_delta);
        ho_delta = calc_entalpy_mix(To+delta, 1.005, 1.805, 4.186, Wi, Wo, 2501.0);
        fn_delta = ho_dry - ho_delta;
        df = (fn_delta-fn)/delta;
        df2 = fn/df;
        To = To - 0.1 * fn/((fn_delta-fn)/delta);
        error = abs(ho_dry - calc_entalpy_mix(To, 1.005, 1.805, 4.186, Wi, Wo, 2501.0));
        iteration = iteration + 1;
    end
else
    disp("凝縮しない")
end

function F = calc_sat_pressure(Ts)
    P_CONVERT = 0.001;
    C1 = -5.6745359e3;
    C2 = 6.3925247;
    C3 = -9.6778430e-3;
    C4 = 6.2215701e-7;
    C5 = 2.0747825e-9;
    C6 = -9.4840240e-13;
    C7 = 4.1635019;
    N1 = 0.11670521452767e4;
    N2 = -0.72421316703206e6;
    N3 = -0.17073846940092e2;
    N4 = 0.12020824702470e5;
    N5 = -0.32325550322333e7;
    N6 = 0.14915108613530e2;
    N7 = -0.4823265731591e4;
    N8 = 0.40511340542057e6;
    N9 = -0.23855557567849e0;
    N10 = 0.65017534844798e3;
    Ts_K = Ts + 273.15;
    
    if Ts < 0.01
        % -100~0.01C//三重点を計算以下は wexler-hyland のシミュレーションプログラム式
        F = exp(C1 / Ts_K + C2 + C3 * Ts_K + C4 * Ts_K^2 + C5 * Ts_K^3 + C6 * Ts_K^4 + C7 * log(Ts_K)) * P_CONVERT;
    else
        %　~647.096K//臨界温度まで一定とするは IAPWS-IF97 実用国際状態式
        alpha = Ts_K + N9 / (Ts_K - N10);
        a2 = alpha * alpha;
        A = a2 + N1 * alpha + N2;
        B = N3 * a2 + N4 * alpha + N5;
        C = N6 * a2 + N7 * alpha + N8;
        F = (2 * C / (-B + (B * B - 4 * A * C)^0.5))^4 / P_CONVERT;
    end 
end

function F = calc_entalpy_air(T,Cpa,Cpv,W,hg0)
    T_K = T+273.15;
    F = Cpa*T_K+W*(Cpv*T_K+hg0);
end

function F = calc_entalpy_mix(T,Cpa,Cpv,Cpw,W1,W2,hg0)
    T_K = T+273.15;
    F = Cpa*T_K+W2*(Cpv*T_K+hg0)+(W1-W2)*Cpw*T_K;
end
