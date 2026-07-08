set(source_files_lib ExampleAnalysisTreeReader.C)
set(header_files_lib_with_classdef ExampleAnalysisTreeReader.hh)
add_project_library(ExampleAnalysisLibrary SOURCES ${source_files_lib} HEADERS_WITH_CLASSDEF ${header_files_lib_with_classdef} LINKDEF Linkdef.h LIBS ExampleAnalysisTree)

set(source_files_exe main.C)
add_project_executable(ExampleAnalysisTreeReader SOURCES ${source_files_exe} LIBS ExampleAnalysisTree ExampleAnalysisLibrary)
