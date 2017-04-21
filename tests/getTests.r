library('RUnit')
 
testSuite = defineTestSuite("checkConvergence",
                              dirs = file.path("./"),
                              testFileRegexp = "^runit.3dgauss.r")
 
testResults <- runTestSuite(testSuite)
 
printTextProtocol(testResults)
