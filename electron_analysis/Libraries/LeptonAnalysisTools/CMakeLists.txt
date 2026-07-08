set(source_files
  LeptonAnalysisTree.C
  LeptonTreePreselection.C
  TrdLikelihoodRatioCalibration.C
  AnalysisSettings.C
  EcalChiSquareLateralCalibration.C
  MeasuringTimeParameterization.C
  ElectronAndPositronFluxModel.C
  EffectiveMeasuringTime.C
)

set(header_files
  LeptonAnalysisTree.hh
  LeptonTreePreselection.hh
  TrdLikelihoodRatioCalibration.hh
  AnalysisSettings.hh
  EcalChiSquareLateralCalibration.hh
  MeasuringTimeParameterization.hh
  Clamping.hh
  ElectronAndPositronFluxModel.hh
  EffectiveMeasuringTime.hh
)

# Work-around for gbatch support, see ACANA-24.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
endif()

# These additional ROOT libraries are required.
set(ADDITIONAL_ROOT_LIBS TreePlayer RooFitCore)

add_project_library(LeptonAnalysisTools SOURCES ${source_files} HEADERS ${header_files} HEADERS_WITH_CLASSDEF ${header_files_with_classdef} LINKDEF Linkdef.h)
target_link_libraries(LeptonAnalysisTools ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} ${ACSOFT_GBATCH_LIBRARY} ${ADDITIONAL_ROOT_LIBS})

