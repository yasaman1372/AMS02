set(source_files ExampleAnalysisTree.C)
set(header_files ExampleAnalysisTree.hh)
add_project_library(ExampleAnalysisTree SOURCES ${source_files} HEADERS ${header_files})
target_link_libraries(ExampleAnalysisTree ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} ${ACSOFT_GBATCH_LIBRARY} TreePlayer)
