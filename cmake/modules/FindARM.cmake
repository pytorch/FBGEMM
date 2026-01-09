
function(get_sve_compiler_flags variable)
  # Look for presence of flags
  check_cxx_compiler_flag(-march=armv8-a+sve COMPILER_SUPPORTS_SVE)
  check_cxx_compiler_flag(-march=armv8-a+sve2 COMPILER_SUPPORTS_SVE2)

  if(COMPILER_SUPPORTS_SVE2)
    BLOCK_PRINT(
      "The compiler supports SVE2 instructions, setting SVE2 compilation flag"
    )
    set(_sve_flags "-march=armv8-a+sve2")
  elseif(COMPILER_SUPPORTS_SVE)
    BLOCK_PRINT(
      "The compiler supports SVE2 instructions, setting SVE2 compilation flag"
    )
    set(_sve_flags "-march=armv8-a+sve")
  endif()

  # Set output variable in parent scope
  set(${variable} ${_sve_flags} PARENT_SCOPE)
endfunction()
