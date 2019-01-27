/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_IMPL_UTILITIES_HPP
#define KOKKOS_CORE_IMPL_UTILITIES_HPP

#include <Kokkos_Macros.hpp>
#include <cstdint>
#include <type_traits>
#include <climits>
#include <limits>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos { namespace Impl {

// same as std::forward
// needed to allow perfect forwarding on the device
template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
T&& forward( typename std::remove_reference<T>::type& arg ) noexcept
{ return static_cast<T&&>(arg); }

template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
T&& forward( typename std::remove_reference<T>::type&& arg ) noexcept
{ return static_cast<T&&>(arg); }

// same as std::move
// needed to allowing moving on the device
template <typename T>
KOKKOS_INLINE_FUNCTION
constexpr
typename std::remove_reference<T>::type&& move( T&& arg ) noexcept
{ return static_cast<typename std::remove_reference<T>::type&&>(arg); }

// empty function to allow expanding a variadic argument pack
template<typename... Args>
KOKKOS_INLINE_FUNCTION
void expand_variadic(Args &&...) {}

//----------------------------------------
// C++14 integer sequence
template< typename T , T ... Ints >
struct integer_sequence {
  using value_type = T ;
  static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

template< typename T , std::size_t N >
struct make_integer_sequence_helper ;

template< typename T , T N >
using make_integer_sequence =
  typename make_integer_sequence_helper<T,N>::type ;

template< typename T >
struct make_integer_sequence_helper< T , 0 >
{ using type = integer_sequence<T> ; };

template< typename T >
struct make_integer_sequence_helper< T , 1 >
{ using type = integer_sequence<T,0> ; };

template< typename T >
struct make_integer_sequence_helper< T , 2 >
{ using type = integer_sequence<T,0,1> ; };

template< typename T >
struct make_integer_sequence_helper< T , 3 >
{ using type = integer_sequence<T,0,1,2> ; };

template< typename T >
struct make_integer_sequence_helper< T , 4 >
{ using type = integer_sequence<T,0,1,2,3> ; };

template< typename T >
struct make_integer_sequence_helper< T , 5 >
{ using type = integer_sequence<T,0,1,2,3,4> ; };

template< typename T >
struct make_integer_sequence_helper< T , 6 >
{ using type = integer_sequence<T,0,1,2,3,4,5> ; };

template< typename T >
struct make_integer_sequence_helper< T , 7 >
{ using type = integer_sequence<T,0,1,2,3,4,5,6> ; };

template< typename T >
struct make_integer_sequence_helper< T , 8 >
{ using type = integer_sequence<T,0,1,2,3,4,5,6,7> ; };

template< typename X , typename Y >
struct make_integer_sequence_concat ;

template< typename T , T ... x , T ... y >
struct make_integer_sequence_concat< integer_sequence<T,x...>
                                   , integer_sequence<T,y...> >
{ using type = integer_sequence< T , x ... , (sizeof...(x)+y)... > ; };

template< typename T , std::size_t N >
struct make_integer_sequence_helper {
  using type = typename make_integer_sequence_concat
    < typename make_integer_sequence_helper< T , N/2 >::type
    , typename make_integer_sequence_helper< T , N - N/2 >::type
    >::type ;
};

//----------------------------------------

template <std::size_t... Indices>
using index_sequence = integer_sequence<std::size_t, Indices...>;

template< std::size_t N >
using make_index_sequence = make_integer_sequence< std::size_t, N>;

//----------------------------------------

template <unsigned I, typename IntegerSequence>
struct integer_sequence_at;

template <unsigned I, typename T, T h0, T... tail>
struct integer_sequence_at<I, integer_sequence<T, h0, tail...> >
  : public integer_sequence_at<I-1u, integer_sequence<T,tail...> >
{
  static_assert( 8 <= I , "Reasoning Error" );
  static_assert( I < integer_sequence<T, h0, tail...>::size(), "Error: Index out of bounds");
};

template < typename T, T h0, T... tail>
struct integer_sequence_at<0u, integer_sequence<T,h0, tail...> >
{
  using type = T;
  static constexpr T value = h0;
};

template < typename T, T h0, T h1, T... tail>
struct integer_sequence_at<1u, integer_sequence<T, h0, h1, tail...> >
{
  using type = T;
  static constexpr T value = h1;
};

template < typename T, T h0, T h1, T h2, T... tail>
struct integer_sequence_at<2u, integer_sequence<T, h0, h1, h2, tail...> >
{
  using type = T;
  static constexpr T value = h2;
};

template < typename T, T h0, T h1, T h2, T h3, T... tail>
struct integer_sequence_at<3u, integer_sequence<T, h0, h1, h2, h3, tail...> >
{
  using type = T;
  static constexpr T value = h3;
};

template < typename T, T h0, T h1, T h2, T h3, T h4, T... tail>
struct integer_sequence_at<4u, integer_sequence<T, h0, h1, h2, h3, h4, tail...> >
{
  using type = T;
  static constexpr T value = h4;
};

template < typename T, T h0, T h1, T h2, T h3, T h4, T h5, T... tail>
struct integer_sequence_at<5u, integer_sequence<T, h0, h1, h2, h3, h4, h5, tail...> >
{
  using type = T;
  static constexpr T value = h5;
};

template < typename T, T h0, T h1, T h2, T h3, T h4, T h5, T h6, T... tail>
struct integer_sequence_at<6u, integer_sequence<T, h0, h1, h2, h3, h4, h5, h6, tail...> >
{
  using type = T;
  static constexpr T value = h6;
};

template < typename T, T h0, T h1, T h2, T h3, T h4, T h5, T h6, T h7, T... tail>
struct integer_sequence_at<7u, integer_sequence<T, h0, h1, h2, h3, h4, h5, h6, h7, tail...> >
{
  using type = T;
  static constexpr T value = h7;
};

//----------------------------------------

template <typename T>
constexpr
T at( const unsigned, integer_sequence<T> ) noexcept
{ return ~static_cast<T>(0); }

template <typename T, T h0, T... tail>
constexpr
T at( const unsigned i, integer_sequence<T, h0> ) noexcept
{ return i==0u ? h0 : ~static_cast<T>(0); }

template <typename T, T h0, T h1>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2, T h3>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2, h3> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 :
         i==3u ? h3 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2, T h3, T h4>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2, h3, h4> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 :
         i==3u ? h3 :
         i==4u ? h4 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2, T h3, T h4, T h5>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2, h3, h4, h5> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 :
         i==3u ? h3 :
         i==4u ? h4 :
         i==5u ? h5 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2, T h3, T h4, T h5, T h6>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2, h3, h4, h5, h6> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 :
         i==3u ? h3 :
         i==4u ? h4 :
         i==5u ? h5 :
         i==6u ? h6 : ~static_cast<T>(0);
}

template <typename T, T h0, T h1, T h2, T h3, T h4, T h5, T h6, T h7, T... tail>
constexpr
T at( const unsigned i, integer_sequence<T, h0, h1, h2, h3, h4, h5, h6, h7, tail...> ) noexcept
{ return i==0u ? h0 :
         i==1u ? h1 :
         i==2u ? h2 :
         i==3u ? h3 :
         i==4u ? h4 :
         i==5u ? h5 :
         i==6u ? h6 :
         i==7u ? h7 : at(i-8u, integer_sequence<T, tail...>{} );
}

//----------------------------------------


template < typename IntegerSequence
         , typename ResultSequence = integer_sequence<typename IntegerSequence::value_type>
         >
struct reverse_integer_sequence_helper;

template <typename T, T h0, T... tail, T... results>
struct reverse_integer_sequence_helper< integer_sequence<T, h0, tail...>, integer_sequence<T, results...> >
  : public reverse_integer_sequence_helper< integer_sequence<T, tail...>, integer_sequence<T, h0, results...> >
{};

template <typename T, T... results>
struct reverse_integer_sequence_helper< integer_sequence<T>, integer_sequence<T, results...> >
{
  using type = integer_sequence<T, results...>;
};


template <typename IntegerSequence>
using reverse_integer_sequence = typename reverse_integer_sequence_helper<IntegerSequence>::type;

//----------------------------------------

template < typename IntegerSequence
         , typename Result
         , typename ResultSequence = integer_sequence<typename IntegerSequence::value_type>
         >
struct exclusive_scan_integer_sequence_helper;

template <typename T, T h0, T... tail, typename Result, T... results>
struct exclusive_scan_integer_sequence_helper
  < integer_sequence<T, h0, tail...>
  , Result
  , integer_sequence<T, results...> >
  : public exclusive_scan_integer_sequence_helper
     < integer_sequence<T, tail...>
     , std::integral_constant<T,Result::value+h0>
     , integer_sequence<T, 0, (results+h0)...> >
{};

template <typename T, typename Result, T... results>
struct exclusive_scan_integer_sequence_helper
  < integer_sequence<T>, Result, integer_sequence<T, results...> >
{
  using type = integer_sequence<T, results...>;
  static constexpr T value = Result::value ;
};

template <typename IntegerSequence>
struct exclusive_scan_integer_sequence
{
  using value_type = typename IntegerSequence::value_type;
  using helper =
    exclusive_scan_integer_sequence_helper
       < reverse_integer_sequence<IntegerSequence>
       , std::integral_constant< value_type , 0 >
       > ;
  using type = typename helper::type ;
  static constexpr value_type value  = helper::value ;
};

//----------------------------------------

template < typename IntegerSequence
         , typename Result
         , typename ResultSequence = integer_sequence<typename IntegerSequence::value_type>
         >
struct inclusive_scan_integer_sequence_helper;

template <typename T, T h0, T... tail, typename Result, T... results>
struct inclusive_scan_integer_sequence_helper
  < integer_sequence<T, h0, tail...>
  , Result
  , integer_sequence<T, results...> >
  : public inclusive_scan_integer_sequence_helper
     < integer_sequence<T, tail...>
     , std::integral_constant<T,Result::value+h0>
     , integer_sequence<T, h0, (results+h0)...> >
{};

template <typename T, typename Result, T... results>
struct inclusive_scan_integer_sequence_helper
  < integer_sequence<T>, Result, integer_sequence<T, results...> >
{
  using type = integer_sequence<T, results...>;
  static constexpr T value = Result::value ;
};

template <typename IntegerSequence>
struct inclusive_scan_integer_sequence
{
  using value_type = typename IntegerSequence::value_type;
  using helper =
    inclusive_scan_integer_sequence_helper
       < reverse_integer_sequence<IntegerSequence>
       , std::integral_constant< value_type , 0 >
       > ;
  using type = typename helper::type ;
  static constexpr value_type value  = helper::value ;
};

template <typename T>
using identity_t = T;

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Add a constant offset to an index sequence
 *
 * The cumbersome partial specialization can be avoided if c++14 is enabled
 *
 * @tparam offset constant offset added to each element of the sequence
 * @tparam class Sequence class to be specialize
 */
/* ----------------------------------------------------------------------------*/
template < std::size_t offset, class > struct AddToSequence {};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Add a constant offset to a Kokkos::Impl::index_sequence
 *
 * Provide a nested Type that define the resulting sequence
 *
 * @tparam offset constant offset added to each element of the sequence
 * @tparam I Input Sequence
 */
/* ----------------------------------------------------------------------------*/
template < std::size_t offset, std::size_t ... I >
struct AddToSequence<offset, Kokkos::Impl::index_sequence<I...>> {
  using Type = Kokkos::Impl::index_sequence<offset+I...>;
};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Get the 2^k, where k is the place of the lowest set bit in i 
 *
 * @tparam T Input integer type
 * @Param i Input integer whose bit pattern is examined
 *
 * @Returns  2^k 
 */
/* ----------------------------------------------------------------------------*/
template < class T > KOKKOS_INLINE_FUNCTION
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type 
lstpowerof2(const T& i) { return i & (~i+1); }

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Count the total number of set bit in the input 
 *
 * @tparam T Input integer type
 * @Param i Input integer whose bit pattern is examined 
 *
 * @Returns  Total number of set bit in i 
 */
/* ----------------------------------------------------------------------------*/
template < class T > KOKKOS_INLINE_FUNCTION
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type 
countSetBits(const T& i) {
  return lstpowerof2(i) ? 1 + countSetBits(i & i-1) : 0; 
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Get 2^k, where k is the index of the place of the j'th (counting
 * from least- to most-significant or right to left) set (non-zero) bit in i 
 *
 * For example, if i == 0b11010101, there are 5 set bits in i,  then
 * powerof2(i,4) == 2^8, powerof2(i,3) == 2^7, powerof2(i,2) == 2^5, 
 * , powerof2(i,1) == 2^2 and powerof2(i,0) == 2^0
 *
 * @tparam T Input integer type
 * @Param i The input integer number whose set bits are examined
 * @Param j Index of the set bit counting from right to left, i.e., the least 
 * significant set bit is 0, the next set bit is 1, ... 
 *
 * @Returns  2^k 
 */
/* ----------------------------------------------------------------------------*/
template < class T > KOKKOS_INLINE_FUNCTION
constexpr typename std::enable_if<std::is_integral<T>::value, T>::type 
powerof2(const T& i, const T& j) {
  return j ? powerof2(i & i-1, j-1) : i & (~i+1);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis Type of input indices to be encoded 
 *
 * For now, all indices assume same type 
 *
 * @tparam nBits Number of bit in the integer index
 *
 */
/* ----------------------------------------------------------------------------*/
template < std::size_t nBits >
using UINT = 
  typename std::conditional<(nBits <= std::numeric_limits<uint8_t>::digits), uint8_t,
  typename std::conditional<(nBits <= std::numeric_limits<uint16_t>::digits), uint16_t,
  typename std::conditional<(nBits <= std::numeric_limits<uint32_t>::digits), uint32_t,
  typename std::conditional<(nBits <= std::numeric_limits<uint64_t>::digits), uint64_t, void>
  ::type>::type>::type>::type;

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Insert some number of 0 between every two neighboring bits in
 * an input integer
 * @tparam T Input integer type
 * @Param i The input integer number 
 * @Param nZeros Number of 0's to be inserted
 * @Param nBits Number of bits of the input integer type
 *
 * @Returns  A 64-bit unsigned integer resulting from inserting 0's 
 */
/* ----------------------------------------------------------------------------*/
template < class T > KOKKOS_INLINE_FUNCTION
constexpr typename std::enable_if<std::is_integral<T>::value, UINT<64>>::type 
insertZeros(const T& i, const T& nZeros, const T& nBits = sizeof(T)*CHAR_BIT) {
  return (i & T{1}) | (nBits == 1 ? 0 : insertZeros<T>(i >> 1, nZeros, nBits-1) << nZeros+1);
}

}} // namespace Kokkos::Impl


#endif //KOKKOS_CORE_IMPL_UTILITIES_HPP

