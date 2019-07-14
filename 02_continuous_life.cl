// This code is derived from:
// https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480
//
// Tested with OpenCL 1.2 CUDA 10.1.152 GeForce GTX 760 on Windows 10

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void continuous_life(write_only image2d_t out, read_only image2d_t image, read_only image2d_t smooth) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const float sigmaW = 2.0f;
  // life parameters
  const float aliveThreshold = .2f;
  const float aliveMin = .125f;
  const float aliveMax = .5f;
  const float birthMin = .25f;
  const float birthMax = .5f;
  
  float life = read_imagef(image, sampler, (int2)(x, y)).s0;
  float aliveNeighborhood = read_imagef(smooth, sampler, (int2)(x, y)).s0;

  // I need to remove the life force at the current position from the neighborhood count, but I'm not sure how to compute the population percentage
  // value of one member of the neighborhood
  // I'm doing an approximation that gaussianW of 1 is approx 3x3 window, and gaussianW of 6 is approx 15x15
  if(life >= aliveThreshold) {
    aliveNeighborhood -= 1.0f / (43.2f * sigmaW - 34.2f);
  }

  // if alive
  if(life >= aliveThreshold) {
    // stay alive and adjust life force if neighborhood alive population is within 12.5% to 50%, with life force being at its peak at 31.25%
    const float h = (aliveMax + aliveMin) / 2.0f;
    const float delta = h - aliveMin;
    if(aliveNeighborhood < aliveMin || aliveNeighborhood > aliveMax) life = 0.0f;
    else life = 1.0f - (fabs(aliveNeighborhood - h) / delta);
  }
  // if almost dead or dead
  else {
    // come to life if neighbohood alive population is withing 25% to 50%, with life force being at its peak at 37.5%
    const float h = (birthMax + birthMin) / 2.0f;
    const float delta = h - birthMin;
    if(aliveNeighborhood < birthMin || aliveNeighborhood > birthMax) life = 0.0f;
    else life = 1.0f - (fabs(aliveNeighborhood - h) / delta);
  }
  
  // convert life to rgb
  life = max(0.0f, min(1.0f, life));
  write_imagef(out, (int2)(x, y), (float4)(life, 0.0f, 0.0f, 0.0f));
}