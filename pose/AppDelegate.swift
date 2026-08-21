//
//  AppDelegate.swift
//  pose
//
//  使用 pre-iOS-13 風格的 UIWindow 啟動流程：
//  Info.plist 不放 UIApplicationSceneManifest，由 AppDelegate 自己掛 window。
//

import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let win = UIWindow(frame: UIScreen.main.bounds)
        win.rootViewController = ViewController()
        win.backgroundColor = .black
        win.makeKeyAndVisible()
        self.window = win
        return true
    }
}
