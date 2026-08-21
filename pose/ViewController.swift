//
//  ViewController.swift
//  pose
//
//  UIKit 容器；實際畫面由 SwiftUI 的 RootView 提供（登入閘門 + 姿勢偵測）。
//

import UIKit
import SwiftUI

class ViewController: UIViewController {

    private var hostingController: UIHostingController<RootView>?

    override func viewDidLoad() {
        super.viewDidLoad()
        edgesForExtendedLayout = .all
        view.backgroundColor = .black

        let hosting = UIHostingController(rootView: RootView())
        hosting.view.backgroundColor = .clear
        if #available(iOS 17.0, *) {
            hosting.safeAreaRegions = [.container]
        }

        addChild(hosting)
        view.addSubview(hosting.view)
        hosting.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hosting.view.topAnchor.constraint(equalTo: view.topAnchor),
            hosting.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            hosting.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hosting.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
        hosting.didMove(toParent: self)
        hostingController = hosting
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        hostingController?.view.frame = view.bounds
    }
}
