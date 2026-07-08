set(source_files
  PhotonTree.C
  IssTree.C
)
set(header_files
  PhotonTree.hh
  IssTree.hh
)

add_project_library(PhotonFramework SOURCES ${source_files} HEADERS ${header_files})
target_link_libraries(PhotonFramework ${ACSOFT_LIBRARIES} ${ACSOFT_AUX_LIBRARIES} ${ACSOFT_GBATCH_LIBRARY} TreePlayer)
