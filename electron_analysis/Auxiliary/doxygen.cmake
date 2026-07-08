find_package(Doxygen)
 
if(DOXYGEN_FOUND)
  file(GLOB doxyfile "Auxiliary/Doxyfile.in")
  configure_file(${doxyfile} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  set_source_files_properties(${doxyfile} PROPERTIES HEADER_FILE_ONLY TRUE)
  
  file(GLOB output_file "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
  set_source_files_properties(${output_file} PROPERTIES HEADER_FILE_ONLY TRUE)
  
  # set source groups for Xcode
  source_group("Template files" FILES ${doxyfile})
  source_group("Generated files" FILES ${output_file})
  add_custom_target(doc
                    ${DOXYGEN_EXECUTABLE} ${output_file}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating API documentation with Doxygen" VERBATIM
                    SOURCES ${output_file} ${doxyfile}
  )
endif()
