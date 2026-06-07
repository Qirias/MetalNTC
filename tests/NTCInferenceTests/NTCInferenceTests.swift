import XCTest
@testable import NTCInference
@testable import NTCAssets

final class NTCInferenceScaffoldTests: XCTestCase {
    func testScaffoldVersionsAgree() {
        XCTAssertEqual(NTCInference.coreVersion, NTCAssets.coreVersion)
    }
}
