set(source_files MCtriggers.C)
add_project_executable(MCtriggers SOURCES ${source_files} LIBS ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} LeptonAnalysisTools)
#add_project_executable(MCtriggers SOURCES ${source_files} HEADERS ${header_files} LIBS ExampleAnalysisTree)
