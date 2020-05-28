/*
 *  Copyright 2008-2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

namespace thrust
{

// NOTE there was a typo in the order of template parameters name here
template <typename ForwardTransform, typename ReverseTransform, typename Iterator>
  class transform_output_iterator;

namespace detail 
{

// Proxy reference that uses Unary Functiont o transform the rhs of assigment
// operator before writing the result to OutputIterator
template <typename ForwardTransform, typename ReverseTransform, typename Iterator>
  class transform_output_iterator_proxy
{
  public:
    __host__ __device__
    transform_output_iterator_proxy(const Iterator& inout, ForwardTransform forward, ReverseTransform reverse) : inout(inout), forward(forward), reverse(reverse)
    {
    }

    __thrust_exec_check_disable__
    template <typename T>
    __host__ __device__
    operator T const() const {
      return forward(*inout);
    }

    __thrust_exec_check_disable__
    template <typename T>
    __host__ __device__
    transform_output_iterator_proxy operator=(const T& x)
    {
      *inout = reverse(x);
      return *this;
    }

    __thrust_exec_check_disable__
    __host__ __device__
    transform_output_iterator_proxy operator=(const transform_output_iterator_proxy& x)
    {
      *inout = reverse(x);
      return *this;
    }

  private:
    Iterator inout;
    ForwardTransform forward;
    ReverseTransform reverse;
};

// Compute the iterator_adaptor instantiation to be used for transform_output_iterator
template <typename ForwardTransform, typename ReverseTransform, typename Iterator>
struct transform_output_iterator_base
{
    typedef thrust::iterator_adaptor
    <
        transform_output_iterator<ForwardTransform, ReverseTransform, Iterator>
      , Iterator
      , typename std::result_of<ForwardTransform(typename std::iterator_traits<Iterator>::value_type)>::type
      , thrust::use_default
      , thrust::use_default
      , transform_output_iterator_proxy<ForwardTransform, ReverseTransform, Iterator>
    > type;
};

// Register trasnform_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
// NOTE there was a typo in the order of template parameters name here
template <typename ForwardTransform, typename ReverseTransform, typename Iterator>
struct is_proxy_reference<
    transform_output_iterator_proxy<ForwardTransform, ReverseTransform, Iterator> >
    : public thrust::detail::true_type {};

} // end detail
} // end thrust

