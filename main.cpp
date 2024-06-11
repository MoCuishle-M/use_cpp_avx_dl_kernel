#include <immintrin.h>
#include <iostream>
#include <memory>
#include <cmath>


//fp32 rmsnorm
// src: {rows, cols}
void RMSnorm_avx(float *src, float *gamma, float *dst, float ln_eps, int rows, int cols) {
  int M = rows;
  int N = cols;

  for (int i = 0; i < M; ++i) {
	// 各个向量求各自的x^2 / len, 最后累加, inter avx512 vec reg
	auto sum_square = _mm512_setzero_ps();
	for (int j = 0; j < N; j += 16) {
	  auto dat = _mm512_loadu_ps(src + i*N + j);
	  // 向量平方
	  auto square = _mm512_mul_ps(dat, dat);
	  sum_square = _mm512_add_ps(sum_square, square);
	}
	float square = _mm512_reduce_add_ps(sum_square);
	float rms = std::sqrt(square/cols+ln_eps);
	auto rms_vec = _mm512_set1_ps(rms);

	for (int j = 0; j < N; j += 16) {
	  //分子除分母
	  //结果由res reg存储到对应offset
	  auto dat = _mm512_loadu_ps(src + i*N + j);
	  auto dst_val = _mm512_mul_ps(_mm512_div_ps(dat, rms_vec),
								   _mm512_loadu_ps(gamma + j)
								   );
	  _mm512_storeu_ps(dst + i*N + j, dst_val);
	}
  }
}


// src: {rows, cols}
// y=(x-E(x))/sqrt(var(x)+eps) * gamma + beta
// var(x) = E(x2)-E(x)2
void layernorm_avx(int M, int N, float *src, float *gamma,
				   float *beta, float *dst, const float &ln_eps) {
  auto len = (float)(N);
  // 返回类型为 __m512 的向量，其中所有元素均设置为零。
  // 512bit，fp32->32bit,一共512/32=16个元素
  auto zero = _mm512_setzero_ps();
  // 将单精度（32 位）浮点值广播到dst的所有元素。
  auto one = _mm512_set1_ps(1.f);
  auto eps = _mm512_set1_ps(ln_eps);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
	// Calculate Mean for each row
	auto sum_mean = _mm512_setzero_ps();
	auto sum_var = _mm512_setzero_ps();
	//Ex和E(X^2)
	for (int j = 0; j < N; j += 16) {
	  // 加载16个fp32
	  auto dat = _mm512_loadu_ps(src + i*N + j);
	  // 向量相加
	  sum_mean = _mm512_add_ps(sum_mean, dat);
	  // 向量对应位置元素相乘，再加到sum_var
	  sum_var = _mm512_fmadd_ps(dat, dat, sum_var);
	}
	// N / 16 个mean和var
	// 16个元素reduce并除以E(x)
	float mean_val = _mm512_reduce_add_ps(sum_mean)/len;
	// E(x^2) - E(x)^2
	float var_val = _mm512_reduce_add_ps(sum_var)/len - mean_val*mean_val;
	auto mean = _mm512_set1_ps(mean_val); // broadcast mean_val to 16x mean_val

	// 1 / sqrt(var + eps)
	auto var = _mm512_div_ps(
		one,
		_mm512_sqrt_ps(_mm512_add_ps(eps,
									 _mm512_max_ps(zero,
												   _mm512_set1_ps(var_val)
												   )
									 )
						 )
		);

	// LayerNorm
	for (int j = 0; j < N; j += 16) {
	  auto amplifier = // (1 / sqrt(var + eps)) * gamma
		  _mm512_mul_ps(_mm512_loadu_ps(gamma + j), var);
	  auto dat = _mm512_loadu_ps(src + i*N + j); // x
	  auto x_mean = _mm512_sub_ps(dat, mean); // x - Ex
	  auto dst_val =
		  _mm512_fmadd_ps(amplifier, x_mean,
						  _mm512_loadu_ps(beta + j));// (x - E(x) / sqrt(var + eps)) * gamma - beta
	  _mm512_storeu_ps(dst + i*N + j, dst_val);
	}
  }
}

//load mask为1的位置，为0的位置pad 0到reg
__m512 _maskz_loadu(const float *data_base, __mmask16 mask) {
  return (__m512)(_mm512_maskz_loadu_ps(mask, (__m512 *)data_base));
}
//store结果到mask为1的位置
void _mask_storeu(float *data_base, __m512 a, __mmask16 mask) {
  _mm512_mask_storeu_ps((__m512 *)data_base, mask, a);
}

// src {m,n}
// bia {n}
void BiasAdd(float *src, float *bias, int m, int n,
			 int stride) {
  for (int i = 0; i < m; ++i) {
	int j = 0;
	for (; j <= n - 16; j += 16) {
	  auto dat = _mm512_loadu_ps(src + i*stride + j);
	  auto bias_dat = _mm512_loadu_ps(bias + j);
	  auto vec_out = _mm512_add_ps(dat, bias_dat);
	  _mm512_storeu_ps(src + i*stride + j, vec_out);
	}
	// 当n不能整除16时，通过mask做padding，把剩余的元素用0 padding到reg即可，算完后，把结果也通过mask存回去
	if (j < n) {
	  // (1 << (n - j)) - 1 用于生成一个二进制数，该数由n-j个连续的1组成.
	  // 接下来的-1操作将上述结果中的最后一个0以及它之后的所有0都转换成了1。
	  // 1000 - 1会变成0111（二进制表示），也就是十进制的7，如果n-j=3，则最终结果就是7（即二进制的111）。
	  // 数据加载是从右到左/低到高放在‘_mm512’寄存器中
	  // MEM[mem_addr+511:mem_addr] := a[511:0]
	  __mmask16 mask = (1 << (n - j)) - 1;
	  auto dat = _maskz_loadu(src + i*stride + j, mask);
	  auto bias_dat = _maskz_loadu(bias + j, mask);
	  auto vec_out = _mm512_add_ps(dat, bias_dat);
	  _mask_storeu(src + i*stride + j, vec_out, mask);
	}
  }
}

void LayNorm_main() {
  const int rows = 100;
  const int cols = 2048;
  // torch中默认为1e-5
  const float epsilon = 1e-5;

  // 使用unique_ptr自动管理内存
  std::unique_ptr<float[]> src(new float[rows*cols]);
  std::unique_ptr<float[]> gamma(new float[cols]);
  std::unique_ptr<float[]> beta(new float[cols]);
  std::unique_ptr<float[]> dst(new float[rows*cols]);

  // 初始化src数组
  for (int i = 0; i < rows*cols; ++i) {
	src[i] = static_cast<float>(i%4 + 1); // 生成周期性序列
  }

  // 初始化gamma和beta数组，仅前cols个元素
  for (int i = 0; i < cols; ++i) {
	gamma[i] = static_cast<float>((i%4 + 1)*0.5);
	beta[i] = static_cast<float>((i%4 + 1)*0.5);
  }

  // 调用Layer Normalization函数
  layernorm_avx(rows, cols, src.get(), gamma.get(), beta.get(), dst.get(), epsilon);

  // 输出处理结果的第一个元素
  std::cout << "LayerNorm Output: " << dst[0] << std::endl;
}

void RMSnorm_main() {
  const int rows = 100;
  const int cols = 2048;
  // torch中默认为1e-6
  const float epsilon = 1e-6;

  // 使用unique_ptr自动管理内存
  std::unique_ptr<float[]> src(new float[rows*cols]);
  std::unique_ptr<float[]> gamma(new float[cols]);
  std::unique_ptr<float[]> beta(new float[cols]);
  std::unique_ptr<float[]> dst(new float[rows*cols]);

  // 初始化src数组
  for (int i = 0; i < rows*cols; ++i) {
	src[i] = static_cast<float>(i%4 + 1); // 生成周期性序列
  }

  // 初始化gamma和beta数组，仅前cols个元素
  for (int i = 0; i < cols; ++i) {
	gamma[i] = 1;
	beta[i] = static_cast<float>((i%4 + 1)*0.5);
  }
  RMSnorm_avx(src.get(), gamma.get(), dst.get(), epsilon, rows, cols);
  std::cout << "rmsnorm output: " << dst[0] << std::endl;
}

void BiasAdd_main() {
  const int rows = 100;
  const int cols = 260; // 设为一个不能整除16的值
  std::unique_ptr<float[]> src(new float[rows*cols]);
  std::unique_ptr<float[]> bias(new float[cols]);

  // initialize
  for (int i = 0; i < rows*cols; i++) {
	src[i] = (float)(i%4 + 1); // 1 2 3 4 1 2 3 4...
	if (i < cols) {
	  bias[i] = 1;
	}
  }
  BiasAdd(src.get(), bias.get(), rows, cols, cols);
  std::cout << "biasadd output: " << src[0] << std::endl; // 1 + 1 = 2
}

int main(int argc, char *argv[]) {
  RMSnorm_main();
  LayNorm_main();
  BiasAdd_main();
}