/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "fbgemm/Assert.h"

using namespace fbgemm;

TEST(AssertTest, CheckPassesOnTrueCondition) {
  FBGEMM_CHECK(true);
  FBGEMM_CHECK(1 == 1);
  FBGEMM_CHECK(2 > 1);
}

TEST(AssertTest, CheckThrowsOnFalseCondition) {
  EXPECT_THROW(FBGEMM_CHECK(false), Error);
  EXPECT_THROW(FBGEMM_CHECK(1 == 2), Error);
  EXPECT_THROW(FBGEMM_CHECK(1 > 2), Error);
}

TEST(AssertTest, CheckThrowsWithDefaultMessage) {
  try {
    FBGEMM_CHECK(false);
    FAIL() << "Expected fbgemm::Error to be thrown";
  } catch (const Error& e) {
    std::string what = e.what();
    EXPECT_NE(what.find("Expected false to be true"), std::string::npos)
        << "Error message should contain the condition. Got: " << what;
  }
}

TEST(AssertTest, CheckThrowsWithCustomMessage) {
  try {
    int x = 42;
    FBGEMM_CHECK(x == 0, "Expected x to be 0, but got ", x);
    FAIL() << "Expected fbgemm::Error to be thrown";
  } catch (const Error& e) {
    std::string what = e.what();
    EXPECT_NE(what.find("Expected x to be 0, but got 42"), std::string::npos)
        << "Error message should contain the custom message. Got: " << what;
  }
}

TEST(AssertTest, CheckMessageContainsSourceLocation) {
  try {
    FBGEMM_CHECK(false);
    FAIL() << "Expected fbgemm::Error to be thrown";
  } catch (const Error& e) {
    std::string what = e.what();
    // Format: [file_name(line:column)] [function_name]: message
    // Should start with [
    EXPECT_EQ(what[0], '[')
        << "Error message should start with [. Got: " << what;
    // Should contain file name
    EXPECT_NE(what.find("AssertTest.cc"), std::string::npos)
        << "Error message should contain file name. Got: " << what;
    // Should contain line and column in format (line:column)]
    EXPECT_NE(what.find("("), std::string::npos)
        << "Error message should contain opening paren. Got: " << what;
    EXPECT_NE(what.find(")]"), std::string::npos)
        << "Error message should contain )] after line:column. Got: " << what;
    // Should contain function name in brackets
    EXPECT_NE(what.find("] ["), std::string::npos)
        << "Error message should contain ] [ between location and function. Got: "
        << what;
    EXPECT_NE(what.find("]: "), std::string::npos)
        << "Error message should contain ]: before message. Got: " << what;
  }
}

TEST(AssertTest, ErrorInheritsFromStdException) {
  try {
    FBGEMM_CHECK(false);
    FAIL() << "Expected exception to be thrown";
  } catch (const std::exception& e) {
    // Should be catchable as std::exception
    EXPECT_NE(std::string(e.what()).size(), 0u);
  }
}

TEST(AssertTest, StrFunctionConcatenatesArgs) {
  EXPECT_EQ(str(), "");

  EXPECT_EQ(str(3.14), "3.14");
  EXPECT_EQ(str(true), "1");
  EXPECT_EQ(str('x'), "x");

  EXPECT_EQ(str("hello"), "hello");
  EXPECT_EQ(str("a", "b", "c"), "abc");
  EXPECT_EQ(str("x = ", 42), "x = 42");
  EXPECT_EQ(str(1, " + ", 2, " = ", 3), "1 + 2 = 3");
}
