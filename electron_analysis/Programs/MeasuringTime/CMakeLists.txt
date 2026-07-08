set(source_files main.C)
add_project_executable(ExampleAnalysisHeMeasuringTime SOURCES ${source_files} HEADERS ${header_files} LIBS ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} LeptonAnalysisTools)
#add_project_executable(ExampleAnalysisHeMeasuringTime SOURCES ${source_files} HEADERS ${header_files} LIBS LeptonTreeWriter)

