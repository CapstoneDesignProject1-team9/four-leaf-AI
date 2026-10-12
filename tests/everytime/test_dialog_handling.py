"""Offline regression tests: no Everytime traffic or saved login profile."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[2] / 'knowledge/crawlers/everytime/find_everytime_lectures.py'
spec = importlib.util.spec_from_file_location('everytime_search_dialog_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DialogTests(unittest.TestCase):
    def test_already_closed_dialog_does_not_escape_callback(self):
        monitor = module.DialogMonitor()
        dialog = Mock(type='alert', message='테스트 알림')
        dialog.dismiss.side_effect = RuntimeError('Protocol error: No dialog is showing')
        monitor.handle(dialog)
        self.assertTrue(monitor.events[0]['already_closed'])
        with self.assertRaises(module.SearchSessionError):
            monitor.check()

    def test_unrelated_failure_is_recorded_and_stops(self):
        monitor = module.DialogMonitor()
        dialog = Mock(type='alert', message='테스트 알림')
        dialog.dismiss.side_effect = RuntimeError('Connection closed')
        monitor.handle(dialog)
        self.assertIn('Connection closed', monitor.events[0]['error'])
        with self.assertRaises(module.SearchSessionError):
            monitor.check()

    def test_no_dialog_can_continue(self):
        module.DialogMonitor().check()

    def test_cleanup_preserves_original_error(self):
        context = Mock()
        context.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaisesRegex(ValueError, 'original failure'):
            try:
                raise ValueError('original failure')
            finally:
                module.close_search_context(context)

    def test_cleanup_error_not_hidden_on_success(self):
        context = Mock()
        context.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            module.close_search_context(context)

    def test_real_local_browser_alert(self):
        from playwright.sync_api import sync_playwright
        monitor = module.DialogMonitor()
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                context.on('dialog', monitor.handle)
                page = context.new_page()
                page.set_content('<p>Offline fixture</p>')
                page.evaluate("alert('local test')")
                self.assertEqual(monitor.events[0]['message'], 'local test')
                self.assertEqual(page.locator('p').inner_text(), 'Offline fixture')
                with self.assertRaises(module.SearchSessionError):
                    monitor.check()
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
