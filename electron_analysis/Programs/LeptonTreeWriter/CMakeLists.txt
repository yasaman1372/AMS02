set(source_files LeptonTreeWriter.C)
add_project_executable(LeptonTreeWriter SOURCES ${source_files} LIBS ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} LeptonAnalysisTools)

