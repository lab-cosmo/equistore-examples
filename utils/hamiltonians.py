###########  I/O UTILITIES ##############
def fix_pyscf_l1(fock, frame, orbs):
    """ pyscf stores l=1 terms in a xyz order, corresponding to (m=0, 1, -1).
        this converts into a canonical form where m is sorted as (-1, 0,1) """
    idx = []
    iorb = 0;
    atoms = list(frame.symbols)
    for atype in atoms:
        cur=()
        for ia, a in enumerate(orbs[atype]):
            n,l,m = a
            if (n,l) != cur:
                if l == 1:
                    idx += [iorb+1, iorb+2, iorb]
                else:
                    idx += range(iorb, iorb+2*l+1)
                iorb += 2*l+1
                cur = (n,l)
    return fock[idx][:,idx]


