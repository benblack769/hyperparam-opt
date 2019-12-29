from math import pi,log

def borehole(xx):
    rw = xx["rad_borehole"]
    r  = xx["radius_influ"]
    Tu = xx["transmissivity_upper"]
    Hu = xx["potentiometric_upper"]
    Tl = xx["transmissivity_lower"]
    Hl = xx["potentiometric_lower"]
    L  = xx["length_borehole"]
    Kw = xx["conductivity"]

    frac1 = 2 * pi * Tu * (Hu-Hl)

    frac2a = 2*L*Tu / (log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = log(r/rw) * (1+frac2a+frac2b)

    y = frac1 / frac2
    return(y)

def low_fidelity_borehole(xx):
    rw = xx["rad_borehole"]
    r  = xx["radius_influ"]
    Tu = xx["transmissivity_upper"]
    Hu = xx["potentiometric_upper"]
    Tl = xx["transmissivity_lower"]
    Hl = xx["potentiometric_lower"]
    L  = xx["length_borehole"]
    Kw = xx["conductivity"]

    frac1 = 5 * Tu * (Hu-Hl)

    frac2a = 2*L*Tu / (log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = log(r/rw) * (1.5+frac2a+frac2b)

    y = frac1 / frac2
    return(y)
