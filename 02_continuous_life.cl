// This code is derived from:
// https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480
//
// Tested with OpenCL 1.2 CUDA 10.1.152 GeForce GTX 760 on Windows 10

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

// see 02_continuous_life_gen_cdf.py
constant float cdf[256] = {4.100286969332956e-05f, 5.640724702971056e-05f, 7.716016989434138e-05f, 0.00010495253081899136f, 0.0001419509353581816f, 0.00019091213471256196f, 0.00025531896972097456f, 0.00033954059472307563f, 0.0004490184655878693f, 0.0005904806312173605f, 0.0007721851579844952f, 0.0010041928617283702f, 0.0012986691435799003f, 0.0016702126013115048f, 0.0021362064871937037f, 0.0027171876281499863f, 0.003437225241214037f, 0.004324298817664385f, 0.005410662852227688f, 0.006733183283358812f, 0.008333628065884113f, 0.010258892551064491f, 0.012561136856675148f, 0.015297814272344112f, 0.018531570211052895f, 0.02232998237013817f, 0.0267651304602623f, 0.03191297873854637f, 0.037852540612220764f, 0.04466484859585762f, 0.05243171378970146f, 0.06123426556587219f, 0.07115131616592407f, 0.08225756883621216f, 0.09462171792984009f, 0.10830442607402802f, 0.12335632741451263f, 0.139816015958786f, 0.15770821273326874f, 0.17704197764396667f, 0.19780932366847992f, 0.21998396515846252f, 0.2435205578804016f, 0.26835429668426514f, 0.2944009006023407f, 0.32155731320381165f, 0.34970253705978394f, 0.3786992132663727f, 0.40839552879333496f, 0.4386276602745056f, 0.46922239661216736f, 0.5f, 0.530777633190155f, 0.5613723397254944f, 0.591604471206665f, 0.6213008165359497f, 0.6502974629402161f, 0.678442656993866f, 0.7055990695953369f, 0.7316457033157349f, 0.7564794421195984f, 0.7800160050392151f, 0.8021906614303589f, 0.8229579925537109f, 0.8422917723655701f, 0.8601839542388916f, 0.8766436576843262f, 0.8916955590248108f, 0.9053782820701599f, 0.9177424311637878f, 0.9288486838340759f, 0.9387657642364502f, 0.9475682973861694f, 0.9553351402282715f, 0.9621474742889404f, 0.9680870175361633f, 0.9732348918914795f, 0.9776700139045715f, 0.9814684391021729f, 0.9847021698951721f, 0.9874388575553894f, 0.9897410869598389f, 0.9916663765907288f, 0.9932668209075928f, 0.9945893287658691f, 0.9956756830215454f, 0.9965627789497375f, 0.9972828030586243f, 0.99786376953125f, 0.998329758644104f, 0.9987013339996338f, 0.9989957809448242f, 0.9992278218269348f, 0.9994094967842102f, 0.999550998210907f, 0.9996604323387146f, 0.9997446537017822f, 0.9998090863227844f, 0.999858021736145f, 0.9998950362205505f, 0.9999228119850159f, 0.9999436140060425f, 0.9999589920043945f, 0.9999703764915466f, 0.9999787211418152f, 0.9999848008155823f, 0.9999892115592957f, 0.9999923706054688f, 0.9999946355819702f, 0.9999962449073792f, 0.9999973773956299f, 0.9999982118606567f, 0.9999987483024597f, 0.9999991655349731f, 0.9999994039535522f, 0.9999996423721313f, 0.9999997615814209f, 0.9999998211860657f, 0.9999998807907104f, 0.9999999403953552f, 0.9999999403953552f, 0.9999999403953552f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

kernel void map_cdf(write_only image2d_t map, read_only image2d_t image) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  float lum = read_imagef(image, sampler, (int2)(x, y)).s0;
  write_imagef(map, (int2)(x, y), (float4)(cdf[(int)(lum*255)], 0.0f, 0.0f, 0.0f));
}

kernel void continuous_life(write_only image2d_t out, read_only image2d_t image, read_only image2d_t smooth) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const float sigmaW = 2.0f;
  // life parameters
  const float aliveThreshold = .2f;
  const float aliveMin = .2f;
  const float aliveMax = .40f;
  const float birthMin = .224f;
  const float birthMax = .26f;
  
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
    // stay alive and adjust life force if neighborhood alive population is within threshold
    const float h = (aliveMax + aliveMin) / 2.0f;
    const float delta = h - aliveMin;
    if(aliveNeighborhood < aliveMin || aliveNeighborhood > aliveMax) life = 0.0f;
    else life = 1.0f - (fabs(aliveNeighborhood - h) / delta);
  }
  // if almost dead or dead
  else {
    // come to life if neighbohood alive population is within threshold
    const float h = (birthMax + birthMin) / 2.0f;
    const float delta = h - birthMin;
    if(aliveNeighborhood < birthMin || aliveNeighborhood > birthMax) life = 0.0f;
    else life = 1.0f - (fabs(aliveNeighborhood - h) / delta);
  }
  
  // convert life to rgb
  life = max(0.0f, min(1.0f, life));
  write_imagef(out, (int2)(x, y), (float4)(life, 0.0f, 0.0f, 0.0f));
}