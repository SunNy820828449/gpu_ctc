#define MIN -1e36
#define MAXMIN -1e30

//kernel set
//(Maxlable_length,1) (batch,2)
__global__ void gpu_forward_backward(float *Output, float *Forward, float *Backward, int *Lable,
	int *Lable_Length, int *Length, int NodeSize, int OutLength, int FDLength, int MaxLable_Length)
{
	int bi = blockIdx.x;
	int bj = blockIdx.y;

	int ti = threadIdx.x;
	//int tj = threadIdx.y;

//the label with blank is no more than 128
	__shared__ float form[128];
	__shared__ float pr[128];

	//the start of batch
	int st = bi*MaxLable_Length;
	int l_l = Lable_Length[bi];
	int l = Length[bi];
	int fst = bi*FDLength;
	int ost = bi*OutLength;

	float cur;
	int label;
	bool mark;
	//load label
	if (ti < l_l)
		label = Lable[st + ti];

	__syncthreads();
	if (bj == 0 && ti < l_l)//forward compute
	{
		//mark compute way
		if (ti > 1 && label != Lable[st + ti - 2])
			mark = true;
		else
			mark = false;

		//initial the first time
		if (ti < 2)
		{
			form[ti] = logf(Output[ost + label]);
			Forward[fst + ti] = form[ti];
		}
		else
		{
			form[ti] = MIN;
			Forward[fst + ti] = 1;
		}
		__syncthreads();

		//loop the form time = 1 to time = T
		for (int t = 1; t < l; t++)
		{
			//load output
			pr[ti] = Output[ost + t*NodeSize + label];//at the time of t
			//compute log value
			if (pr[ti] == 0)
				pr[ti] = MIN;
			else
				pr[ti] = logf(pr[ti]);

			//if the rest time length < the label length 
			if (ti < (l_l - 2 * (l - t)))//斜率是 2 因此下边成立
				cur = MIN;
			else
			{
				//the first label
				if (ti == 0)
					cur = pr[ti] + form[ti]; //logf(Output[ost + t*NodeSize + lable[ti]]) + form[ti];
				else
				{
					//compute the value
					if (mark)
						cur = pr[ti] + //logf(Output[ost + t*NodeSize + lable[ti]])
						+Active::LogExchangeFunction(form[ti - 2],
						Active::LogExchangeFunction(form[ti - 1], form[ti]));
					else
						cur = pr[ti] + //logf(Output[ost + t*NodeSize + lable[ti]])
						+Active::LogExchangeFunction(form[ti - 1], form[ti]);
				}
			}
			//
			__syncthreads();
			//if the value is small 
			if (cur > MAXMIN)
			{
				Forward[fst + t*l_l + ti] = cur;
				form[ti] = cur;
			}
			else
			{
				Forward[fst + t*l_l + ti] = 1;
				form[ti] = MIN;
			}

			__syncthreads();
		}
	}
	else if (ti < l_l)//backward
	{
		//mark compute the no blank and no repet
		if (ti < l_l - 2 && label != Lable[st + ti + 2])
			mark = true;
		else
			mark = false;
		//initial the last time
		if (ti >= l_l - 2)
		{
			form[ti] = 0;
			Backward[fst + (l - 1)*l_l + ti] = 0;
		}
		else
		{
			form[ti] = MIN;
			Backward[fst + (l - 1)*l_l + ti] = 1;
		}

		__syncthreads();

		//loop from time T-1 to the first
		for (int t = l - 2; t >= 0; t--)
		{
			//load output
			pr[ti] = Output[ost + (t + 1)*NodeSize + label];//the next time

			if (pr[ti] == 0)
				pr[ti] = MIN;
			else
				pr[ti] = logf(pr[ti]);
			__syncthreads();

			if (ti > 2 * t + 1)
				cur = MIN;
			else
			{
				if (ti == l_l - 1)
					cur = pr[ti] +
					form[ti];
				else
				{
					if (mark)
						cur = Active::LogExchangeFunction(Active::LogExchangeFunction(
						pr[ti + 1] + form[ti + 1],
						pr[ti + 2] + form[ti + 2]),
						pr[ti] + form[ti]);
					else
						cur = Active::LogExchangeFunction(
						pr[ti + 1] + form[ti + 1],
						pr[ti] + form[ti]);
				}
			}
			__syncthreads();

			if (cur > MAXMIN)
			{
				Backward[fst + t*l_l + ti] = cur;
				form[ti] = cur;
			}
			else
			{
				Backward[fst + t*l_l + ti] = 1;
				form[ti] = MIN;
			}

			__syncthreads();
		}
	}
}
