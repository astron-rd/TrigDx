#include <iomanip>
#include <iostream>
#include <vector>

#include <trigdx/trigdx.hpp>

int main() {
  constexpr float pi = 3.14159265358979323846f;
  const std::vector<float> angles = {0.0f,      pi / 6.0f, pi / 4.0f,
                                     pi / 3.0f, pi / 2.0f, pi};

  std::vector<float> sin_values(angles.size());
  std::vector<float> cos_values(angles.size());

  LookupBackend<16384> backend;
  backend.init();
  backend.compute_sincosf(angles.size(), angles.data(), sin_values.data(),
                          cos_values.data());

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "angle(rad)\tsin(x)\t\tcos(x)\n";

  for (std::size_t i = 0; i < angles.size(); ++i) {
    std::cout << angles[i] << '\t' << sin_values[i] << '\t' << cos_values[i]
              << '\n';
  }

  return 0;
}
