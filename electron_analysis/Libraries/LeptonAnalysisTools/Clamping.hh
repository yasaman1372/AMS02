#ifndef Clamping_hh
#define Clamping_hh

#include <cassert>
#include <limits>

#include "OutputHandler.hh"

template<typename TargetType, typename SourceType>
TargetType safeClampTo(const SourceType& value, const TargetType& defaultValue) {

  assert(sizeof(SourceType) >= sizeof(TargetType));
  static TargetType gMin = std::numeric_limits<TargetType>::min();
  static TargetType gMax = std::numeric_limits<TargetType>::max();

  // Perform comparision in 64bit, to assure that the value clampTo can be used to clamp from an uint to an int.
  // Without using the up-cast to 64bit uint max does NOT fit into the int range, and the comparision would fail.
  if (Long64_t(value) >= Long64_t(gMin) && Long64_t(value) <= Long64_t(gMax))
    return TargetType(value);
  return defaultValue;
}

template<typename TargetType, typename SourceType>
TargetType clampTo(const SourceType& value) {

  assert(sizeof(SourceType) >= sizeof(TargetType));
  static TargetType gMin = std::numeric_limits<TargetType>::min();
  static TargetType gMax = std::numeric_limits<TargetType>::max();

  // Perform comparision in 64bit, to assure that the value clampTo can be used to clamp from an uint to an int.
  // Without using the up-cast to 64bit uint max does NOT fit into the int range, and the comparision would fail.
  if (Long64_t(value) < Long64_t(gMin)) {
    Utilities::OutputHandler::FatalHandler(__FILE__, __LINE__, __PRETTY_FUNCTION__, 0, Utilities::OutputHandler::EverywhereWithEvent)
      << "Passed value=" << Long64_t(value) << " (original=" << value << ") is smaller than minimum value for "
      << "desired type, min=" << Long64_t(gMin) << " (original=" << gMin << "). Aborting!" << std::endl;
  }

  if (Long64_t(value) > Long64_t(gMax)) {
    Utilities::OutputHandler::FatalHandler(__FILE__, __LINE__, __PRETTY_FUNCTION__, 0, Utilities::OutputHandler::EverywhereWithEvent)
      << "Passed value=" << Long64_t(value) << " (original=" << value << ") is larger than maximum value for "
      << "desired type, max=" << Long64_t(gMax) << " (original=" << gMax << "). Aborting!" << std::endl;
  }

  return TargetType(value);
}

#endif // Clamping_hh

