import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import gazebo_world_generator.gazebo_world_generator as main_mod

def test_parse_arguments():
    with patch.object(sys, 'argv', ['prog', '--description', 'test world', '--output', 'test.sdf']):
        args = main_mod.parse_arguments()
        assert args.description == 'test world'
        assert str(args.output) == 'test.sdf'

def test_console_handler_emit():
    handler = main_mod.ConsoleHandler()
    record = MagicMock()
    record.levelno = 20 # INFO
    record.msg = "Test message"
    record.args = ()
    record.getMessage.return_value = "Test message"
    record.exc_info = None
    
    # We just want to ensure it doesn't crash when emitting
    with patch('builtins.print'):
        handler.emit(record)

def test_setup_logging():
    with patch.object(main_mod.logger, 'addHandler') as mock_add:
        main_mod.setup_logging(debug=True)
        assert mock_add.called
