file(GLOB_RECURSE header_files ${ROVER_INCLUDE_PATH}/Rover/Python/*.hpp)
file(GLOB_RECURSE source_files ${ROVER_SOURCE_PATH}/Python/*.cpp)
add_library(python SHARED ${header_files} ${source_files})
set_source_files_properties(${header_files} PROPERTIES HEADER_FILE_ONLY TRUE)
set_target_properties(python PROPERTIES OUTPUT_NAME rover)
target_compile_definitions(python PRIVATE ROVER_BUILD_DLL)
target_include_directories(python SYSTEM PRIVATE
  ${PYBIND11_INCLUDE_PATH} ${PYTHON_INCLUDE_PATH})
set_target_properties(python PROPERTIES CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN 0)
if(MSVC)
  set_target_properties(python PROPERTIES SUFFIX ".pyd")
endif()
target_compile_options(python PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fvisibility=hidden>)
if(UNIX)
  set_target_properties(python PROPERTIES PREFIX "")
endif()
target_link_libraries(python PRIVATE
  $<$<CONFIG:Debug>:${PYTHON_LIBRARY_DEBUG_PATH}>
  $<$<NOT:$<CONFIG:Debug>>:${PYTHON_LIBRARY_OPTIMIZED_PATH}>)
install(TARGETS python CONFIGURATIONS Debug
  DESTINATION ${LIB_INSTALL_DIRECTORY}/Debug)
install(TARGETS python CONFIGURATIONS Release RelWithDebInfo
  DESTINATION ${LIB_INSTALL_DIRECTORY}/Release)
