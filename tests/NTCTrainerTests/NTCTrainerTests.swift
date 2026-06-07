import XCTest
@testable import NTCTrainer
@testable import NTCAssets

final class NTCTrainerScaffoldTests: XCTestCase {
    func testScaffoldVersionsAgree() {
        XCTAssertEqual(NTCTrainer.coreVersion, NTCAssets.coreVersion)
    }
}
