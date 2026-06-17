file(GLOB_RECURSE source_files ${ROVER_SOURCE_PATH}/Tests/*.cpp)
add_executable(rover_tester ${source_files})
target_include_directories(rover_tester SYSTEM PRIVATE ${DOCTEST_INCLUDE_PATH})
if(UNIX)
  target_link_libraries(rover_tester PRIVATE pthread)
endif()
add_custom_command(TARGET rover_tester POST_BUILD COMMAND rover_tester)
install(TARGETS rover_tester CONFIGURATIONS Debug
  DESTINATION ${TEST_INSTALL_DIRECTORY}/Debug)
install(TARGETS rover_tester CONFIGURATIONS Release RelWithDebInfo
  DESTINATION ${TEST_INSTALL_DIRECTORY}/Release)
