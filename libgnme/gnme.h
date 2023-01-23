#ifndef LIBGNME_H
#define LIBGNME_H

/** \mainpage Generalised Non-Orthogonal Matrix Elements 
 
    This library provides a general framework for computing generalised 
    non-orthogonal matrix elements in electronic structure methods.

    The methods used in this library are summarised in
    - ["Generalised Nonorthogonal Matrix Elements: Unifying Wick's Theorem and the Slater--Condon Rules"](https://arxiv.org/abs/2101.10944);
      H. G. A. Burton, arxiv:2101.10944 (2021)

    \ref gnme "Components of the GNME library"

    \authors Hugh Burton 2022-
 **/

/** \defgroup gnme GNME library
  
    The GNME framework consists of two main interfaces:
    - \ref gnme_slater_condon "Implementation of the Generalised Slater-Condon rules"
    - \ref gnme_wick "Implement of the Extended Nonorthogonal Wick's Theorem"
 **/

/** \defgroup gnme_slater_condon Generalised Slater-Condon Rules
    \ingroup gnme
 **/

/** \defgroup gnme_wick Extended Non-Orthogonal Wick's Theorem
    \ingroup gnme
 **/

/** \defgroup gnme_utils Utilities
    \ingroup gnme
 **/

#endif // LIBGNME_H
