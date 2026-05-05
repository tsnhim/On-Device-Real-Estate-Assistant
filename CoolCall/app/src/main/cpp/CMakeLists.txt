cmake_minimum_required(VERSION 3.22.1)
project("coolcall")

# 1. Fetch SentencePiece source
include(FetchContent)
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.1.99
)
FetchContent_MakeAvailable(sentencepiece)

# 2. Build your bridge
add_library(coolcall SHARED tokenizer.cpp)
target_include_directories(coolcall PRIVATE ${sentencepiece_SOURCE_DIR}/src)

# 3. Link dependencies
target_link_libraries(sentencepiece log)

target_link_libraries(coolcall
    sentencepiece-static
    log
)
