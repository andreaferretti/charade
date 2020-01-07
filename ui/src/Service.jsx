import React from 'react'
import Grid from '@material-ui/core/Grid'
import Typography from '@material-ui/core/Typography'
import Button from '@material-ui/core/Button'
import Switch from '@material-ui/core/Switch'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardActions from '@material-ui/core/CardActions'
import Divider from '@material-ui/core/Divider'
import Paper from '@material-ui/core/Paper'
import Popover from '@material-ui/core/Popover'
import { styled } from '@material-ui/core/styles'
import ReactJson from 'react-json-view'

import theme from './theme'

const Note = styled(Grid)({
  'font-size': '0.7em',
  'color': theme.palette.primary.dark
})

const Item = styled(Grid)({
  padding: theme.spacing(1)
})

const SpacedArea = styled(CardActions)({
  padding: theme.spacing(2)
})

const SpacedPaper = styled(Paper)({
  padding: theme.spacing(5)
})

class Service extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      open: false
    }
  }

  closePopover() {
    this.setState({ open: false })
  }

  openPopover() {
    this.setState({ open: true })
  }

  onSelect() {
    this.props.onSelect(
      this.props.desc.task,
      this.props.desc.name,
      this.props.desc.langs,
      this.props.desc['extra-params']
    )
  }

  render() {
    const dependencies = this.props.desc.deps.length ? this.props.desc.deps.join(', ') : '-'
    const languages = this.props.desc.langs ? this.props.desc.langs.join(', ') : '*'

    return (
      <Item item xs={3}>
        <Card>
          <CardContent>
            <Typography variant='overline'>{this.props.desc.name}</Typography>

            <Grid container>
              <Note item xs={6}>
                Languages: <br />
                <Typography variant='subtitle2'>{languages}</Typography>
              </Note>
              <Divider orientation='vertical' />
              <Note item xs={5}>
                Deps: <br/>
                <Typography variant='subtitle2'>{dependencies}</Typography>
              </Note>
            </Grid>

          </CardContent>

          <SpacedArea>
            <Switch
              checked={this.props.selected}
              onChange={this.onSelect.bind(this)}>
            </Switch>
            <Button
              variant='contained'
              color='primary'
              onClick={this.openPopover.bind(this)}
            >+</Button>
            <Popover
              open={this.state.open}
              onClose={this.closePopover.bind(this)}
              anchorOrigin={{
                vertical: 'center',
                horizontal: 'center',
              }}
              transformOrigin={{
                vertical: 'center',
                horizontal: 'center',
              }}
            >
              <SpacedPaper>
                <Typography variant='subtitle2'>{this.props.desc.task} - {this.props.desc.name}</Typography>
                <ReactJson
                  src={this.props.desc}
                  displayObjectSize={false}
                  displayDataTypes={false}
                />
              </SpacedPaper>
            </Popover>
          </SpacedArea>
        </Card>
      </Item>
    )
    }
}

export default Service