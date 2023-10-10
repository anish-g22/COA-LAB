#ifndef COUNTER_H_INCLUDED
#define COUNTER_H_INCLUDED

extern int cycle_cnt, waiting, XMEM, XALU, Others, Issued;
extern int w_ch, w_sc, w_dw, i_m, i_sp, i_int, i_dp, i_sfu, i_tc, i_spec;
extern int x_spint, x_dp, x_sfu, x_tc, x_spec, o_ibe, o_wb;
extern int w_mi, xalu_mi, xmem_mi, o_mi;
extern int invalid;

#endif