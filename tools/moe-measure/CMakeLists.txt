set(TARGET llama-moe-measure-common)
add_library(${TARGET} STATIC moe-measure-common.cpp moe-measure-media.cpp)
target_link_libraries(${TARGET} PUBLIC llama-common llama)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

add_executable(llama-moe-measure moe-measure.cpp)
target_link_libraries(llama-moe-measure PRIVATE llama-moe-measure-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-moe-measure PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS llama-moe-measure RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    add_executable(test-moe-measure-common moe-measure-common-tests.cpp)
    target_link_libraries(test-moe-measure-common PRIVATE llama-moe-measure-common)
    target_compile_features(test-moe-measure-common PRIVATE cxx_std_17)
    add_test(NAME test-moe-measure-common COMMAND test-moe-measure-common)

    add_executable(test-moe-measure-media moe-measure-media-tests.cpp)
    target_link_libraries(test-moe-measure-media PRIVATE llama-moe-measure-common)
    target_compile_features(test-moe-measure-media PRIVATE cxx_std_17)
    add_test(NAME test-moe-measure-media COMMAND test-moe-measure-media)

    add_executable(test-moe-measure moe-measure-tests.cpp)
    target_link_libraries(test-moe-measure PRIVATE llama-moe-measure-common mtmd ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(test-moe-measure PRIVATE cxx_std_17)
    add_test(NAME test-moe-measure COMMAND test-moe-measure)

    add_executable(test-moe-measure-cli moe-measure-cli-tests.cpp)
    target_compile_features(test-moe-measure-cli PRIVATE cxx_std_17)
    add_dependencies(test-moe-measure-cli llama-moe-measure)
    add_test(NAME test-moe-measure-cli
             COMMAND test-moe-measure-cli $<TARGET_FILE:llama-moe-measure>)
endif()
