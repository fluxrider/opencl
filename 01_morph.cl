// This code is derived from:
// https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void dilate(__read_only image2d_t in, __write_only image2d_t out) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  // find max in 3x3 region
  float extreme = 0;
  for(int i = -1; i <= 1; i++) {
    for(int j = -1; j <= 1; j++) {
      const float pixel = read_imagef(in, sampler, (int2)(x + i, y + j)).s0;
      extreme = max(extreme, pixel);
    }
  }

  write_imagef(out, (int2)(x, y), (float4)(extreme, 0.0f, 0.0f, 0.0f));
}