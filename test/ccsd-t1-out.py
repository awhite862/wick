

def get_t1(t1, t2, h1e, h2e):
    t1_vo   = t1.transpose()
    t2_vvoo = t2.transpose(2, 3, 1, 0)
    h1e_vo  = h1e.vo
    h2e_vvoo = h2e.vvoo
    h2e_ovov = h2e.ovov

    # T1 part
    T1 += 1.0*einsum('ai->ai', h1e_vo)
    T1 += -1.0*einsum('ji,aj->ai', h1e_oo, t1_vo)
    T1 += 1.0*einsum('ab,bi->ai', h1e_vv, t1_vo)
    T1 += -1.0*einsum('jb,abji->ai', h1e_ov, t2_vvoo)
    T1 += -1.0*einsum('jaib,bj->ai', h2e_ovov, t1_vo)
    T1 += 0.5*einsum('jkib,abkj->ai', h2e_ooov, t2_vvoo)
    T1 += -0.5*einsum('jabc,cbji->ai', h2e_ovvv, t2_vvoo)
    T1 += -1.0*einsum('jb,bi,aj->ai', h1e_ov, t1_vo, t1_vo)
    T1 += -1.0*einsum('jkib,aj,bk->ai', h2e_ooov, t1_vo, t1_vo)
    T1 += 1.0*einsum('jabc,ci,bj->ai', h2e_ovvv, t1_vo, t1_vo)
    T1 += -0.5*einsum('jkbc,aj,cbki->ai', h2e_oovv, t1_vo, t2_vvoo)
    T1 += -0.5*einsum('jkbc,ci,abkj->ai', h2e_oovv, t1_vo, t2_vvoo)
    T1 += 1.0*einsum('jkbc,cj,abki->ai', h2e_oovv, t1_vo, t2_vvoo)
    T1 += 1.0*einsum('jkbc,ci,aj,bk->ai', h2e_oovv, t1_vo, t1_vo, t1_vo)

    # T2 part
    T2 += 1.0*einsum('ai->ai', h1e_vo)
    T2 += -1.0*einsum('ji,aj->ai', h1e_oo, t1_vo)
    T2 += 1.0*einsum('ab,bi->ai', h1e_vv, t1_vo)
    T2 += -1.0*einsum('jb,abji->ai', h1e_ov, t2_vvoo)
    T2 += -1.0*einsum('jaib,bj->ai', h2e_ovov, t1_vo)
    T2 += 0.5*einsum('jkib,abkj->ai', h2e_ooov, t2_vvoo)
    T2 += -0.5*einsum('jabc,cbji->ai', h2e_ovvv, t2_vvoo)
    T2 += -1.0*einsum('jb,bi,aj->ai', h1e_ov, t1_vo, t1_vo)
    T2 += -1.0*einsum('jkib,aj,bk->ai', h2e_ooov, t1_vo, t1_vo)
    T2 += 1.0*einsum('jabc,ci,bj->ai', h2e_ovvv, t1_vo, t1_vo)
    T2 += -0.5*einsum('jkbc,aj,cbki->ai', h2e_oovv, t1_vo, t2_vvoo)
    T2 += -0.5*einsum('jkbc,ci,abkj->ai', h2e_oovv, t1_vo, t2_vvoo)
    T2 += 1.0*einsum('jkbc,cj,abki->ai', h2e_oovv, t1_vo, t2_vvoo)
    T2 += 1.0*einsum('jkbc,ci,aj,bk->ai', h2e_oovv, t1_vo, t1_vo, t1_vo)